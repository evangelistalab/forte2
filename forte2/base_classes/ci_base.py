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
      ``nroot`` and ``make_rdm(root, order=..., spin_type=...)``.

    Subclasses implement the eigensolve and RDM primitives (``run``,
    ``reset_eigensolver``, ``get_convergence_status``, ``make_rdm``).
    """

    log_level: int = logger.VERBOSITY_DEBUG

    ### Non-init attributes
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    # By default, assume the solver is not invariant to orbital rotations.
    # Subclasses can override this.
    orbital_rotation_invariant: ClassVar[bool] = False

    # Capability declarations. Every concrete solver states which RDM/cumulant orders and
    # representations it supports, and which orders accept cross-state (transition)
    # requests. None means "never declared" and is reported as a NotImplementedError naming
    # the class; an empty tuple is a valid declaration meaning "not supported".
    _rdm_orders: ClassVar[tuple[int, ...] | None] = None
    _rdm_spin_types: ClassVar[tuple[str, ...] | None] = None
    _rdm_cross_state_orders: ClassVar[tuple[int, ...] | None] = None
    _cumulant_orders: ClassVar[tuple[int, ...] | None] = None
    _cumulant_spin_types: ClassVar[tuple[str, ...] | None] = None

    def __call__(self, parent_method):
        self._register_parent_method(parent_method)
        if not hasattr(self, "final_orbitals"):
            self.final_orbitals = getattr(parent_method, "final_orbitals", "original")
        return self

    def reset(self):
        """Invalidate this solver, forcing _solve() to rebuild sub_solvers on
        the next run() instead of resolving in the (stale) current basis."""
        self.first_run = True
        return super().reset()

    def _print_energy_summary(self) -> None:
        """
        Print this solver's energy table. Overridden by solvers that report more
        than one energy, e.g. selected CI's variational and PT2-corrected values.
        Called by the driver that owns this solver.
        """
        from forte2.ci.ci_utils import pretty_print_ci_summary

        pretty_print_ci_summary(self.sa_info, self.evals_per_solver)

    def _transition_property_energies(self):
        """The energies the transition-property table is printed against."""
        return self.evals_per_solver

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

    def _validate_rdm_inputs(
        self,
        left_root,
        right_root,
        order,
        allowed_orders,
        spin_type,
        allowed_spin_types,
        cross_state_orders,
    ):
        """
        Validate a state-averaged `make_rdm`/`make_cumulant` request: order and spin type
        against the solver's allowed sets, and (if `left_root`/`right_root` resolve to
        different states) that the requested order is one that supports cross-state
        (transition) requests and that the two states have compatible electron counts.
        Returns the resolved states and per-state roots, plus the canonical spin type.
        """
        # Defer import to avoid a circular import at module load
        from forte2.ci.ci_utils import check_capability_declared, normalize_spin_type

        check_capability_declared(
            self,
            orders=allowed_orders,
            spin_types=allowed_spin_types,
            cross_state_orders=cross_state_orders,
        )
        if order not in allowed_orders:
            raise ValueError(f"order must be one of {allowed_orders}, got {order}.")
        spin_type = normalize_spin_type(spin_type.lower())
        if spin_type not in allowed_spin_types:
            raise ValueError(
                f"spin_type must be one of {allowed_spin_types}, got '{spin_type}'."
            )

        left_state, left_root_in_state = self._get_state_root(left_root)
        if right_root is not None:
            right_state, right_root_in_state = self._get_state_root(right_root)
        else:
            right_state = left_state
            right_root_in_state = left_root_in_state

        if left_state != right_state:
            if order not in cross_state_orders:
                raise ValueError(
                    f"Cross-state requests are not supported for order {order}. Got left_root "
                    f"in state {left_state} and right_root in state {right_state}."
                )
            if (
                self.sa_info.states[left_state].na
                != self.sa_info.states[right_state].na
                or self.sa_info.states[left_state].nb
                != self.sa_info.states[right_state].nb
            ):
                raise ValueError(
                    "Cross-state RDMs are only supported for states with the same number of alpha and beta electrons."
                )

        return (
            left_state,
            right_state,
            left_root_in_state,
            right_root_in_state,
            spin_type,
        )

    def compute_average_energy(self):
        """
        Compute the average energy from the CI roots using the weights.

        Returns
        -------
        float
            Average energy of the CI roots.
        """
        return np.dot(self.weights_flat, self.evals_flat)

    def make_average_rdm(self, order: int):
        """
        Make the state-averaged RDM of the given order, from the CI vectors, in each
        sub-solver's native representation (spin-free for one-component backends,
        spin-orbital for two-component backends).

        Parameters
        ----------
        order : int
            The RDM order. Availability depends on the backend (e.g. selected CI caps at 2).

        Returns
        -------
        NDArray
            The state-averaged RDM.
        """
        spin_type = "so" if self.two_component else "sf"
        rdm = np.zeros((self.norb,) * (2 * order), dtype=self.dtype)
        for i, ci_solver in enumerate(self.sub_solvers):
            for j in range(ci_solver.nroot):
                rdm += ci_solver.make_rdm(j, None, order=order, spin_type=spin_type) * (
                    self.sa_info.weights[i][j]
                )
        return rdm

    def make_average_cumulant(self, order: int):
        """
        Make the state-averaged cumulant of the given order, computed from the state-averaged
        RDMs (cumulants are non-linear in the RDMs, so this cannot be a weighted sum of
        per-root cumulants).

        Parameters
        ----------
        order : int
            The cumulant order (2 or 3).

        Returns
        -------
        NDArray
            The state-averaged cumulant.
        """
        # Defer import to avoid a circular import at module load
        from forte2.ci.ci_utils import (
            make_2cumulant_sf,
            make_2cumulant_so,
            make_3cumulant_sf,
            make_3cumulant_so,
        )

        if order not in (2, 3):
            raise ValueError(f"order must be one of (2, 3), got {order}.")

        dm1 = self.make_average_rdm(1)
        dm2 = self.make_average_rdm(2)
        if order == 2:
            return (make_2cumulant_so if self.two_component else make_2cumulant_sf)(
                dm1, dm2
            )
        dm3 = self.make_average_rdm(3)
        return (make_3cumulant_so if self.two_component else make_3cumulant_sf)(
            dm1, dm2, dm3
        )

    def make_active_space_ints(self):
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

        ints = self.make_active_space_ints()

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
            g1_avg = self.make_average_rdm(1)
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
        rdm_spin_type = "so" if self.two_component else "sf"
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
            rdm = self.sub_solvers[istate].make_rdm(
                iroot_in_state, None, order=1, spin_type=rdm_spin_type
            )
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
                    tdm = self.make_rdm(_ici, _jci, order=1, spin_type=rdm_spin_type)
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

    def make_cumulant(self, root: int, *, order: int, spin_type: str):
        """
        Make the cumulant of the given order and representation for one absolute CI root.

        Parameters
        ----------
        root : int
            the absolute CI root.
        order : int
            The cumulant order (2 or 3, depending on the backend).
        spin_type : str
            "sf" (spin-free) or "so" (spin-orbital), depending on the backend. The
            aliases "spin_free", "spin-free", "spin_orbital", "spin-orbital", and
            "spinorbital" are also accepted.

        Returns
        -------
        NDArray
            The cumulant.
        """
        state, _, root_in_state, _, spin_type = self._validate_rdm_inputs(
            root,
            None,
            order,
            self._cumulant_orders,
            spin_type,
            self._cumulant_spin_types,
            # a cumulant is defined for a single state, never between two
            (),
        )
        return self.sub_solvers[state].make_cumulant(
            root_in_state, order=order, spin_type=spin_type
        )

    @abstractmethod
    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: int,
        spin_type: str,
    ): ...


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
