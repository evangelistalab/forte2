from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .active_space_solver import ActiveSpaceSolver, RelActiveSpaceSolver


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

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def _collect_child_kwargs(self, target_cls):
        """Collect keyword arguments for child solvers."""
        # Defer import to avoid polluting top-level namespace
        from dataclasses import fields as _dc_fields

        # Take all init fields of the target dataclass and copy values from `self` if present
        names = {f.name for f in _dc_fields(target_cls) if f.init}
        return {n: getattr(self, n) for n in names if hasattr(self, n)}

    def _startup(self):
        super()._startup()

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
                self.sa_info.states[left_state].na != self.sa_info.states[right_state].na
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
                rdm1 += ci_solver.make_1rdm(j) * self.weights[i][j]
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
                rdm2 += ci_solver.make_2rdm(j) * self.weights[i][j]

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
                rdm3 += ci_solver.make_3rdm(j) * self.weights[i][j]

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

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def reset_eigensolver(self): ...

    @abstractmethod
    def get_convergence_status(self): ...

    @abstractmethod
    def make_1rdm(self, left_root: int, right_root: int | None = None): ...

    @abstractmethod
    def make_2rdm(self, left_root: int, right_root: int | None = None): ...


@dataclass
class RelCIBase(RelActiveSpaceSolver):
    ### Non-init attributes
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    __call__ = CIBase.__call__
    _collect_child_kwargs = CIBase._collect_child_kwargs
    _get_state_root = CIBase._get_state_root
    _validate_rdm_inputs = CIBase._validate_rdm_inputs
    compute_average_energy = CIBase.compute_average_energy
    make_average_1rdm = CIBase.make_average_1rdm
    make_average_2rdm = CIBase.make_average_2rdm
    make_average_3rdm = CIBase.make_average_3rdm
    make_average_2cumulant = CIBase.make_average_2cumulant
    make_average_3cumulant = CIBase.make_average_3cumulant
    make_average_cumulants = CIBase.make_average_cumulants

    def _startup(self):
        super()._startup(two_component=True)
        if not self.system.two_component:
            raise ValueError(
                "RelCISolver requires a two-component system. Please use a parent method that can provide a two-component wavefunction."
            )

    @abstractmethod
    def run(self, use_asym_ints=False): ...

    @abstractmethod
    def reset_eigensolver(self): ...

    @abstractmethod
    def get_convergence_status(self): ...

    @abstractmethod
    def make_1rdm(self, left_root: int, right_root: int | None = None): ...

    @abstractmethod
    def make_2rdm(self, left_root: int, right_root: int | None = None): ...
