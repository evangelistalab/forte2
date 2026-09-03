from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .active_space_solver import ActiveSpaceSolver, RelActiveSpaceSolver
from forte2.helpers import logger


@dataclass
class CIBase(ActiveSpaceSolver):
    """
    Base class for (state-averaged) CI-type active-space solvers.

    Provides the representation-agnostic orchestration shared by all concrete
    solvers: root/state bookkeeping, state-averaged RDMs, and cumulants. These
    methods assume the subclass populates the following in its ``_startup``/``run``:

    - ``self.norb`` : number of active orbitals
    - ``self.evals_flat`` : flat array of eigenvalues over all roots
    - ``self.sub_solvers`` : list of per-state worker objects, each exposing
      ``nroot`` and ``make_{1,2,3}rdm(root)`` returning the spin-free RDMs.

    Subclasses implement the eigensolve and RDM primitives (``run``,
    ``reset_eigensolver``, ``get_convergence_status``, ``make_sf_*``/``make_sd_*``).
    """

    ### Non-init attributes
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    # By default, assume the solver is not invariant to orbital rotations.
    # Subclasses can override this.
    orbital_rotation_invariant: ClassVar[bool] = False

    def __call__(self, parent_method):
        self._register_parent_method(parent_method)
        return self

    def reset(self):
        """Invalidate this solver, forcing _solve() to rebuild sub_solvers on
        the next run() instead of resolving in the (stale) current basis."""
        self.first_run = True
        return super().reset()

    @property
    def ci_solver(self):
        """
        A bare CI solver is itself the ci_solver, matching MCOptimizerBase.ci_solver
        so that downstream methods (e.g. DSRG) can treat both parent kinds uniformly.
        """
        return self

    def _collect_child_kwargs(self, target_cls):
        """Collect keyword arguments for child solvers."""
        # Defer import to avoid polluting top-level namespace
        from dataclasses import fields as _dc_fields

        # Take all init fields of the target dataclass and copy values from `self` if present
        names = {f.name for f in _dc_fields(target_cls) if f.init}
        return {n: getattr(self, n) for n in names if hasattr(self, n)}

    def _get_state_root(self, absolute_root) -> tuple[int, int]:
        if absolute_root < 0 or absolute_root >= self.sa_info.nroots_sum:
            raise ValueError(
                f"absolute_root must be between 0 and {self.sa_info.nroots_sum - 1}, but got {absolute_root}."
            )
        return self.sa_info.absolute_root_map[absolute_root]

    def _validate_rdm_inputs(self, left_root, right_root):
        left_state, left_root_in_state = self._get_state_root(left_root)
        if right_root is not None:
            right_state, right_root_in_state = self._get_state_root(right_root)
        else:
            right_state = left_state
            right_root_in_state = left_root_in_state

        if left_state != right_state:
            if (
                self.sa_info.states[left_state].na
                != self.sa_info.states[right_state].na
                or self.sa_info.states[left_state].nb
                != self.sa_info.states[right_state].nb
            ):
                raise ValueError(
                    "Cross-state RDMs are only supported for states with the same number of alpha and beta electrons."
                )

        return left_state, right_state, left_root_in_state, right_root_in_state

    def compute_average_energy(self):
        """
        Compute the average energy from the CI roots using the weights.

        Returns
        -------
        float
            Average energy of the CI roots.
        """
        return np.dot(self.weights_flat, self.evals_flat)

    def make_average_1rdm(self):
        """
        Make the average spin-free one-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free one-particle RDM.
        """
        rdm1 = np.zeros((self.norb,) * 2, dtype=self.dtype)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm1 += ci_solver.make_1rdm(j) * self.sa_info.weights[i][j]
        return rdm1

    def make_average_2rdm(self):
        """
        Make the average spin-free two-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free two-particle RDM.
        """
        rdm2 = np.zeros((self.norb,) * 4, dtype=self.dtype)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm2 += ci_solver.make_2rdm(j) * self.sa_info.weights[i][j]

        return rdm2

    def make_average_3rdm(self):
        """
        Make the average spin-free three-particle RDM from the CI vectors.

        Returns
        -------
        NDArray
            Average spin-free three-particle RDM.
        """
        rdm3 = np.zeros((self.norb,) * 6, dtype=self.dtype)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm3 += ci_solver.make_3rdm(j) * self.sa_info.weights[i][j]

        return rdm3

    def make_average_2cumulant(self):
        # Defer import to avoid a circular import at module load
        from forte2.ci.ci_utils import make_2cumulant_sf, make_2cumulant_so

        dm1 = self.make_average_1rdm()
        dm2 = self.make_average_2rdm()
        if self.two_component:
            return make_2cumulant_so(dm1, dm2)
        else:
            return make_2cumulant_sf(dm1, dm2)

    def make_average_3cumulant(self):
        # Defer import to avoid a circular import at module load
        from forte2.ci.ci_utils import make_3cumulant_sf, make_3cumulant_so

        dm1 = self.make_average_1rdm()
        dm2 = self.make_average_2rdm()
        dm3 = self.make_average_3rdm()
        if self.two_component:
            return make_3cumulant_so(dm1, dm2, dm3)
        else:
            return make_3cumulant_sf(dm1, dm2, dm3)

    def make_average_cumulants(self):
        # Defer import to avoid a circular import at module load
        from forte2.ci.ci_utils import (
            make_2cumulant_sf,
            make_2cumulant_so,
            make_3cumulant_sf,
            make_3cumulant_so,
        )

        dm1 = self.make_average_1rdm()
        dm2 = self.make_average_2rdm()
        dm3 = self.make_average_3rdm()
        if self.two_component:
            lambda2 = make_2cumulant_so(dm1, dm2)
            lambda3 = make_3cumulant_so(dm1, dm2, dm3)
        else:
            lambda2 = make_2cumulant_sf(dm1, dm2)
            lambda3 = make_3cumulant_sf(dm1, dm2, dm3)
        return dm1, dm2, lambda2, lambda3

    def _make_active_space_ints(self):
        """Build the active-space integrals for the current ``self.mos``."""
        return self._integrals_cls(
            self.system,
            self.mos.C[0],
            self.active_indices,
            self.core_indices,
        )

    def _extra_worker_kwargs(self, index, state):
        """Hook for per-state worker kwargs beyond the shared ones."""
        return {}

    def _startup(self):
        super()._startup()

        self.norb = self.mo_space.nactv
        # no distinction between core and frozen core in the CI solver
        self.core_indices = (
            self.mo_space.frozen_core_indices + self.mo_space.core_indices
        )
        self.active_indices = self.mo_space.active_indices

        ints = self._make_active_space_ints()

        active_orbsym = [
            [self.mos.irrep_indices[0][i] for i in active_space]
            for active_space in self.mo_space.active_orbitals
        ]

        self.sub_solvers = []
        for i, state in enumerate(self.sa_info.states):
            # one worker per state / GAS restriction
            kwargs = self._collect_child_kwargs(self._ss_solver_cls)
            # needed by the worker but not present as attributes of the solver
            kwargs.update(
                {
                    "ints": ints,
                    "state": state,
                    "nroot": self.sa_info.nroots[i],
                    "active_orbsym": active_orbsym,
                }
            )
            kwargs.update(self._extra_worker_kwargs(i, state))
            self.sub_solvers.append(self._ss_solver_cls(**kwargs))

    def _collect_root_results(self):
        """Hook for subclasses to gather extra per-solver results during ``run``."""

    def _solve(self):
        """
        Solve the state-averaged CI problem in the *current* orbital basis.
        This method can be called repeatedly.
        """
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evals_per_solver = []
        for ci_solver in self.sub_solvers:
            ci_solver.run()
            self.evals_per_solver.append(ci_solver.evals)

        self.evals_flat = np.concatenate(self.evals_per_solver)
        self.E_avg = self.compute_average_energy()
        self.E = self.evals_flat

        self._collect_root_results()

        self.executed = True
        return self

    def run(self):
        """Solve in the current basis. Single-shot solvers extend this."""
        return self._solve()

    def _rotate_final_orbitals(self, final_orbitals):
        """
        Apply the requested ``final_orbitals`` transformation.

        Parameters
        ----------
        final_orbitals : str
            Type of final orbitals.
        """
        from forte2.orbitals import (
            check_final_orbital_energy_invariance,
            make_final_orbitals,
        )

        irrep_indices = np.asarray(self.mos.irrep_indices[0], dtype=int)[
            self.mo_space.orig_to_contig
        ]
        C_contig = self.mos.C[0][:, self.mo_space.orig_to_contig].copy()
        g1_act = (
            self.make_average_1rdm()
            if final_orbitals != "original"
            else None
        )
        C_final = make_final_orbitals(
            final_orbitals,
            system=self.system,
            mo_space=self.mo_space,
            irrep_indices=irrep_indices,
            C_contig=C_contig,
            g1_act=g1_act,
        )
        if final_orbitals == "original":
            return

        # undo the contiguous ordering
        self.mos.C[0] = C_final[:, self.mo_space.contig_to_orig].copy()

        old_E = self.E.copy()
        old_E_avg = self.E_avg

        # re-solve the CI in the final orbital basis
        ints = self._make_active_space_ints()
        self.set_ints(ints.E, ints.H, ints.V)
        # reset_eigensolver() drops the DavidsonLiuSolver, so the rerun rebuilds it
        # from davidson_liu_params (including its maxiter).
        self.reset_eigensolver()
        self._solve()

        check_final_orbital_energy_invariance(
            hard_fail=self.orbital_rotation_invariant,
            tol=self._final_orbital_energy_tol,
            old_E=old_E,
            new_E=self.E,
            old_E_avg=old_E_avg,
            new_E_avg=self.E_avg,
            hard_fail_hint="Consider increasing davidson_liu_params.maxiter.",
        )

    def set_ints(self, scalar, oei, tei):
        """
        Set the active-space integrals for every sub-solver.

        Parameters
        ----------
        scalar : float
            The scalar energy term.
        oei : NDArray
            One-electron active-space integrals in the MO basis.
        tei : NDArray
            Two-electron active-space integrals in the MO basis.
        """
        for ci_solver in self.sub_solvers:
            ci_solver.set_ints(scalar, oei, tei)

    def reset_eigensolver(self):
        """
        Reset the eigensolver for each sub-solver.

        This forces a re-initialization of the eigensolver in the next run, and
        also forces re-computation of the guess vectors. Useful whenever the
        integrals have changed (e.g. after semi-canonicalization).
        """
        for ci_solver in self.sub_solvers:
            ci_solver.reset_eigensolver()

    def get_convergence_status(self):
        """
        Get the convergence status of each sub-solver.

        Returns
        -------
        list[bool]
            A list of booleans indicating whether each sub-solver has converged.
        """
        status = []
        for ci_solver in self.sub_solvers:
            if ci_solver.eigensolver is None:
                # Exact diagonalization
                status.append(True)
            else:
                status.append(ci_solver.eigensolver.converged)
        return status

    def get_top_determinants(self, n=5):
        """
        Get the top `n` determinants for each root based on their coefficients
        in the CI vector.

        Parameters
        ----------
        n : int, optional, default=5
            The number of top determinants to return.

        Returns
        -------
        top_dets : list[list[tuple[Determinant, float]]]
            ``top_dets[i]`` contains a list of (Determinant, coefficient) tuples
            for the `i`-th root.
        """
        top_dets = []
        for ci_solver in self.sub_solvers:
            top_dets += ci_solver.get_top_determinants(n)
        return top_dets

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers for the CI states and store them
        in ``self.nat_occs`` (this method returns None).
        If more than one CI roots are requested, then self.nat_occs_avg stores
        the state-averaged natural occupation numbers.

        """
        nos = []
        for ci_solver in self.sub_solvers:
            nos.append(ci_solver.compute_natural_occupation_numbers())
        self.nat_occs = np.concatenate(nos, axis=1)
        self.nat_occs_avg = None
        if self.sa_info.nroots_sum > 1:
            g1_avg = self.make_average_1rdm()
            self.nat_occs_avg = np.linalg.eigvalsh(g1_avg)[::-1]

    def compute_transition_properties(self, C=None):
        """
        Compute the transition dipole moments, oscillator strengths, and vertical
        transition energies from the 1-TDMs.

        Parameters
        ----------
        C : NDArray, optional
            The MO coefficients. If not provided, ``self.mos.C[0]`` is used.

        Returns
        -------
        transition_dipoles : dict[tuple[int, int], NDArray]
            Maps pairs of CI roots (absolute_root_i, absolute_root_j) to their
            transition dipole moments. Also saved in ``self.transition_dipoles``.
        oscillator_strengths : dict[tuple[int, int], float]
            Maps pairs of CI roots to their oscillator strengths. Also saved in
            ``self.oscillator_strengths``.
        vertical_transition_energies : dict[tuple[int, int], float]
            Maps pairs of CI roots to their vertical transition energies. Also
            saved in ``self.vertical_transition_energies``.
        """
        from forte2.props import get_1e_property

        if not self.executed:
            raise RuntimeError("CI solver has not been executed yet.")

        if C is None:
            C = self.mos.C[0]

        Cact = C[:, self.active_indices]
        Ccore = C[:, self.core_indices]
        # spin-summed 1-RDM for the spatial-orbital case; spinors are singly occupied
        factor = 1.0 if self.two_component else 2.0
        rdm_core = factor * np.einsum("pi,qi->pq", Ccore, Ccore.conj(), optimize=True)
        # this includes nuclear dipole contribution
        core_dip = get_1e_property(
            self.system, rdm_core, property_name="dipole", unit="au"
        )
        self.transition_dipoles = OrderedDict()
        self.oscillator_strengths = OrderedDict()
        self.vertical_transition_energies = OrderedDict()
        for ici in range(self.sa_info.nroots_sum):
            istate, iroot_in_state = self._get_state_root(ici)
            rdm = self.sub_solvers[istate].make_1rdm(iroot_in_state)
            # Different (back-)transformation rules for RDMs:
            # O_{mu}^{nu} = C_{mu}^p <phi_p|O|phi^q> C^q_{nu} = C^H O[mo] C
            # rdm^{mu}_{nu} = C^{mu}_p <a^p a_q> C^q_{nu} = C^* rdm[mo] C^T
            rdm = np.einsum("ij,pi,qj->pq", rdm, Cact.conj(), Cact, optimize=True)
            dip = get_1e_property(
                self.system, rdm, property_name="electric_dipole", unit="au"
            )
            self.transition_dipoles[(ici, ici)] = dip + core_dip
            # No oscillator strength or vertical transition energy for i->i transitions
            self.oscillator_strengths[(ici, ici)] = 0.0
            self.vertical_transition_energies[(ici, ici)] = 0.0
            for jci in range(ici + 1, self.sa_info.nroots_sum):
                jstate, jroot_in_state = self._get_state_root(jci)
                try:
                    vte = (
                        self.evals_per_solver[jstate][jroot_in_state]
                        - self.evals_per_solver[istate][iroot_in_state]
                    )
                    # Reverse the order of states for negative VTE to ensure the
                    # transition dipole is always computed from lower to higher state.
                    if vte < 0:
                        _ici, _jci = jci, ici
                        vte = -vte
                    else:
                        _ici, _jci = ici, jci
                    tdm = self.make_1rdm(_ici, _jci)
                    tdm = np.einsum(
                        "ij,pi,qj->pq", tdm, Cact.conj(), Cact, optimize=True
                    )
                    tdip = get_1e_property(
                        self.system, tdm, property_name="electric_dipole", unit="au"
                    )
                    self.transition_dipoles[(_ici, _jci)] = tdip
                    self.oscillator_strengths[(_ici, _jci)] = (
                        (2 / 3) * vte * np.linalg.norm(tdip) ** 2
                    )
                    self.vertical_transition_energies[(_ici, _jci)] = vte
                except Exception as e:
                    # ValueError: non-relativistic CI where the two states have
                    #   different na/nb, so cross-state RDMs are unsupported.
                    # NotImplementedError: two-component CI, cross-state RDMs are
                    #   not implemented yet.
                    logger.log_warning(
                        f"Transition properties between CI states {_ici} and {_jci} cannot be computed. Original exception: {e}"
                    )
                    continue

        return (
            self.transition_dipoles,
            self.oscillator_strengths,
            self.vertical_transition_energies,
        )

    @abstractmethod
    def make_1rdm(self, left_root: int, right_root: int | None = None): ...

    @abstractmethod
    def make_2rdm(self, left_root: int, right_root: int | None = None): ...


@dataclass
class RelCIBase(RelActiveSpaceSolver, CIBase):
    """
    Two-component counterpart of :class:`CIBase`.
    """

    def _startup(self):
        super()._startup()
        if not self.system.two_component:
            raise ValueError(
                "RelCISolver requires a two-component system. Please use a parent method that can provide a two-component wavefunction."
            )
