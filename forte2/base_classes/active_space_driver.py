from typing import ClassVar

import numpy as np


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
        self.ci_solver.log_level = max(
            logger.get_verbosity_level(), logger.VERBOSITY_WARNING
        )
        self.ci_solver.die_if_not_converged = self.die_if_not_converged
        return self

    def run(self):
        """
        Solve with the owned solver, then finish: copy its results,
        rotate the final orbitals once, and report.
        """
        if not self.parent_method.executed:
            self.parent_method.run()
        self.ci_solver = self.ci_solver(self.parent_method)
        self.ci_solver.run()

        self.system = self.ci_solver.system
        self.mos = self.ci_solver.mos
        self.mo_space = self.ci_solver.mo_space
        self.dtype = self.ci_solver.dtype

        self._rotate_final_orbitals()
        self._post_process()
        self.executed = True
        return self

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

    def _rotate_final_orbitals(self) -> None:
        """
        Rotate to the requested final orbitals, once, at the end of the driver's
        run, then re-solve in the new basis. A no-op beyond copying the solver's
        results unless "semicanonical" or "natural" was asked for.

        Changing the orbital basis is the driver's job, not the solver's: the
        solver only ever answers in the basis it is handed. Here the driver picks
        the new orbitals, hands the solver integrals built from them, and asks it
        to solve again.
        """
        from forte2.orbitals import (
            check_final_orbital_energy_invariance,
            make_final_orbitals,
        )

        # the solver shares this driver's MO object, so it sees the new orbitals
        self.ci_solver.mos = self.mos
        self._copy_solver_results()
        if self.final_orbitals not in ("semicanonical", "natural"):
            return

        irrep_indices = np.asarray(self.mos.irrep_indices[0], dtype=int)[
            self.mo_space.orig_to_contig
        ]
        C_contig = self.mos.C[0][:, self.mo_space.orig_to_contig].copy()
        C_final = make_final_orbitals(
            self.final_orbitals,
            system=self.system,
            mo_space=self.mo_space,
            irrep_indices=irrep_indices,
            C_contig=C_contig,
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
        # reset_eigensolver() drops the DavidsonLiuSolver, so the rerun rebuilds it
        # from davidson_liu_params (including its maxiter).
        self.ci_solver.reset_eigensolver()
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

    def _print_orbital_composition(self) -> None:
        """Hook: print the composition of the final orbitals. No-op by default."""

    def _post_process(self) -> None:
        from forte2.ci.ci_utils import (
            pretty_print_ci_dets,
            pretty_print_ci_nat_occ_numbers,
            pretty_print_ci_transition_props,
        )

        self.ci_solver._print_energy_summary()
        self.ci_solver.compute_natural_occupation_numbers()
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
            self.ci_solver.compute_transition_properties(self.mos.C[0])
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
