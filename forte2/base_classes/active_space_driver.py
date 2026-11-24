from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .mo import MO
from .results import ActiveSpaceResult


class ActiveSpaceDriver:
    """
    Shared behavior for methods that own an active-space solver and use it to
    finish a calculation, e.g., :class:`MCOptimizer` and  :class:`CI`.

    A driver runs its solver, copies the solver's results onto itself,
    rotates the final orbitals once, and reports.
    """

    # Tolerance for the sanity check that a final-orbital rotation landed on the same
    # solution it started from.
    _final_orbital_energy_tol: ClassVar[float] = 1e-8

    def __post_init__(self):
        from forte2.base_classes.ci_base import CIBase, RelCIBase
        from forte2.orbitals import validate_final_orbitals

        if not isinstance(self.ci_solver, (CIBase, RelCIBase)):
            raise ValueError("ci_solver must be an instance of CIBase or RelCIBase.")
        validate_final_orbitals(self.final_orbitals)

        self.requires = {"system", "mos"}
        self.provides = {"system", "mos", "mo_space"}
        self.requires_attrs |= self.ci_solver.requires_attrs

    def __call__(self, parent_method):
        from forte2.helpers import logger

        self._register_parent_method(parent_method)
        self.ci_solver.log_level = logger.get_verbosity_level()
        # the driver raises after storing its result, so the solver must not
        self.ci_solver.die_if_not_converged = False
        return self

    def run(self):
        """
        Solve with the owned solver, then finish: copy its results,
        rotate the final orbitals once, report, and store the result.
        """
        if not self.parent_method.executed:
            self.parent_method.run()
        self.ci_solver = self.ci_solver(self.parent_method)
        self.ci_solver.run()

        self.system = self.ci_solver.system
        self.mos = self.ci_solver.mos
        self.mo_space = self.ci_solver.mo_space
        self.dtype = self.ci_solver.dtype

        self._finalize(
            all(self.ci_solver.get_convergence_status()),
            "CI solver did not converge for all roots.",
        )
        self.executed = True
        return self

    def _finalize(self, converged: bool, message: str) -> None:
        """
        Rotate the final orbitals, report, and store ``self.result``. A run that
        dies skips the rotation and raises only after the result is stored.
        """
        from forte2.system import ModelSystem

        die = self.die_if_not_converged and not converged
        mo_unrotated, active_rotation = self._rotate_final_orbitals(skip=die)
        # every root's 1-RDM is built once, for both the report and the result
        rdm1 = self.ci_solver._make_root_rdm1s()
        dipoles = None
        if not isinstance(self.system, ModelSystem):
            dipoles = self.ci_solver.compute_dipoles(self.mos.C[0], rdm1)
        self._post_process(rdm1, dipoles)
        self.result = self._make_result(mo_unrotated, active_rotation, rdm1, dipoles)
        if die:
            raise RuntimeError(message)

    def reset(self):
        """
        Invalidate this driver and the solver it owns before a new run().
        """
        self.ci_solver.reset()
        return super().reset()

    def _copy_solver_results(self) -> None:
        """
        Copy the owned solver's energies onto the driver. ``E_ci`` is a copy, so
        a later re-solve of the solver cannot mutate it behind the driver's back.
        """
        self.E_ci = np.array(self.ci_solver.E)
        self.E_avg = self.ci_solver.E_avg
        self.E = self.E_avg

    def _rotate_final_orbitals(
        self, skip: bool = False
    ) -> tuple[MO, NDArray] | tuple[None, None]:
        """
        Rotate to the requested final orbitals, once, at the end of the driver's
        run, then re-solve in the new basis. A no-op beyond copying the solver's
        results if ``skip`` is True or unless "semicanonical" or "natural" was asked for.

        Changing the orbital basis is the driver's job, not the solver's: the
        solver only ever answers in the basis it is handed. Here the driver picks
        the new orbitals, hands the solver integrals built from them, and asks it
        to solve again.

        Returns
        -------
        mo_unrotated : MO | None
            A copy of the orbitals before the rotation.
        Uactv : NDArray | None
            The active-space rotation, in active-index order.
            Both are None if the orbitals were not rotated.
        """
        from forte2.orbitals import (
            check_final_orbital_energy_invariance,
            make_final_orbitals,
        )
        from forte2.orbitals.ci_rotation import rotate_ci_vectors

        # the solver shares this driver's MO object, so it sees the new orbitals
        self.ci_solver.mos = self.mos
        self._copy_solver_results()
        if skip or self.final_orbitals not in ("semicanonical", "natural"):
            return None, None

        mo_unrotated = self.mos.copy()
        contig = self.mos.permute(self.mo_space.orig_to_contig)
        C_final, Uactv = make_final_orbitals(
            self.final_orbitals,
            system=self.system,
            mo_space=self.mo_space,
            irrep_indices=np.asarray(contig.irrep_indices[0], dtype=int),
            C_contig=contig.C[0],
            g1_act=self.make_average_rdm(1),
        )
        # undo the contiguous ordering
        self.mos.C[0] = C_final[:, self.mo_space.contig_to_orig].copy()

        old_E_ci = self.E_ci.copy()
        old_E_avg = self.E_avg

        # re-solve in the final orbital basis, the same way MCSCF re-solves after an
        # orbital step: new integrals, then a plain run of the solver
        ints = self.ci_solver.make_active_space_ints()
        self.ci_solver.set_ints(ints.E, ints.H, ints.V)
        guesses = None
        if self.ci_solver.orbital_rotation_invariant:
            # rotated vectors are the converged solution in the new basis
            guesses = rotate_ci_vectors(self.ci_solver, Uactv)
        self.ci_solver.reset_eigensolver(guesses=guesses)
        self.ci_solver.run()
        self._copy_solver_results()

        check_final_orbital_energy_invariance(
            hard_fail=self.ci_solver.orbital_rotation_invariant,
            tol=self._final_orbital_energy_tol,
            old_E=old_E_ci,
            new_E=self.E_ci,
            old_E_avg=old_E_avg,
            new_E_avg=self.E_avg,
            hard_fail_hint="Consider increasing davidson_liu_params.maxiter.",
        )
        return mo_unrotated, Uactv

    def _make_result(
        self,
        mo_unrotated: MO | None,
        active_rotation: NDArray | None,
        rdm1: NDArray,
        dipoles: NDArray | None,
    ) -> ActiveSpaceResult:
        ci = self.ci_solver
        transition_properties = {}
        if self.do_transition_dipole:
            transition_properties = {
                "trdm1": dict(ci.transition_rdms),
                "transition_dipoles": dict(ci.transition_dipoles),
                "oscillator_strengths": dict(ci.oscillator_strengths),
            }

        return ActiveSpaceResult(
            ci_energies=np.array(ci.E_tot),
            mo_coeff=self.mos.copy(),
            mo_coeff_unrotated=mo_unrotated,
            active_rotation=active_rotation,
            rdm1=rdm1,
            dipoles=dipoles,
            **transition_properties,
        )

    def _print_orbital_composition(self) -> None:
        """Hook: print the composition of the final orbitals. No-op by default."""

    def _post_process(self, rdm1: NDArray, dipoles: NDArray | None) -> None:
        from forte2.ci.ci_utils import (
            pretty_print_ci_dets,
            pretty_print_ci_nat_occ_numbers,
            pretty_print_ci_transition_props,
        )

        self.ci_solver._print_energy_summary()
        self.ci_solver.compute_natural_occupation_numbers(rdm1)
        pretty_print_ci_nat_occ_numbers(
            self.ci_solver.sa_info,
            self.mo_space,
            self.ci_solver.nat_occs,
            getattr(self.ci_solver, "nat_occs_avg", None),
        )
        top_dets = self.ci_solver.get_top_determinants()
        pretty_print_ci_dets(self.ci_solver.sa_info, self.mo_space, top_dets)
        self._print_orbital_composition()
        if self.do_transition_dipole:
            self.ci_solver.compute_transition_properties(self.mos.C[0], dipoles)
            pretty_print_ci_transition_props(
                self.ci_solver.sa_info,
                self.ci_solver.transition_dipoles,
                self.ci_solver.oscillator_strengths,
                self.ci_solver._transition_property_energies(),
            )

    ### Reduced density matrices and cumulants, forwarded to the owned solver.
    ### Everything else the solver exposes is reached through ``ci_solver``.

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: int,
        spin_type: str,
    ):
        return self.ci_solver.make_rdm(
            left_root, right_root, order=order, spin_type=spin_type
        )

    def make_cumulant(self, root: int, *, order: int, spin_type: str):
        return self.ci_solver.make_cumulant(root, order=order, spin_type=spin_type)

    def make_average_rdm(self, order: int):
        return self.ci_solver.make_average_rdm(order)

    def make_average_cumulant(self, order: int):
        return self.ci_solver.make_average_cumulant(order)
