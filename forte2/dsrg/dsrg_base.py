from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from forte2.base_classes import SystemMixin, MOsMixin, MOSpaceMixin, ActiveSpaceSolver
from forte2.ci import CISolver
from forte2.helpers import logger


@dataclass
class DSRGBase(SystemMixin, MOsMixin, MOSpaceMixin, ABC):
    """Base class for DSRG methods."""

    ci_solver: CISolver
    flow_param: float = 0.5

    # Reference relaxation options
    relax_reference: int | str | bool = False
    relax_maxiter: int = 10
    relax_tol: float = 1e-6

    # Non-init attributes
    executed: bool = field(init=False, default=False)
    converged: bool = field(init=False, default=False)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def __post_init__(self):
        # parse reference relaxation options
        if isinstance(self.relax_reference, bool):
            self.nrelax = self.relax_maxiter if self.relax_reference else 0
        elif isinstance(self.relax_reference, int):
            assert self.relax_reference >= 0, "relax_reference must be non-negative."
            self.nrelax = min(self.relax_reference, self.relax_maxiter)
        elif isinstance(self.relax_reference, str):
            assert self.relax_reference.lower() in [
                "once",
                "twice",
                "iterate",
            ], "relax_reference must be one of 'once', 'twice', or 'iterate'."
            if self.relax_reference.lower() == "once":
                self.nrelax = 1
            elif self.relax_reference.lower() == "twice":
                self.nrelax = 2
            else:
                self.nrelax = self.relax_maxiter
            # [Edsrg(fixed_reference), Edsrg(relaxed_reference), Eref]
            self.relax_energies = np.zeros((self.nrelax, 3))
        else:
            logger.log_warning(
                "Reference relaxation options not recognized, no relaxation will be performed."
            )
            self.nrelax = 0
            self.relax_energies = np.zeros((1, 3))

    def _startup(self):
        assert isinstance(self.parent_method, ActiveSpaceSolver)
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        self.two_component = self.system.two_component

        MOSpaceMixin.copy_from_upstream(self, self.parent_method)
        perm = self.mo_space.orig_to_contig

        MOsMixin.copy_from_upstream(self, self.parent_method)
        self._C = self.C[0][:, perm].copy()

        self.ints, self.cumulants = self.get_integrals()

        # only initialize the CI solver if reference relaxation is requested
        # initialized in do_reference_relaxation()
        self.ci_solver = None

    def run(self):
        self._startup()

        self.E_fixed_ref = self.solve_dsrg()
        self.relax_energies[0, 0] = self.E_fixed_ref
        self.relax_energies[0, 2] = self.parent_method.E
        self.E = self.E_fixed_ref

        for irelax in range(self.nrelax):
            # "twice": DSRG -> relax -> DSRG -> done
            if irelax == 2 and self.relax_reference == "twice":
                self.converged = True
                break

            self.E_relaxed_ref = self.do_reference_relaxation()
            self.E = self.E_relaxed_ref
            self.relax_energies[irelax, 1] = self.E_relaxed_ref
            self.relax_energies[irelax, 2] = self.ci_solver.E

            # "once": DSRG -> relax -> done
            if self.relax_reference == "once":
                self.converged = True
                break

            self.converged = self.test_relaxation_convergence(irelax)
            if self.converged:
                break

            self.solve_dsrg()
        else:
            logger.log_warning(
                f"DSRG reference relaxation did not converge in {self.nrelax} iterations."
            )
        self.executed = True
        return self

    def test_relaxation_convergence(self, irelax):
        if irelax == 0 or self.relax_reference != "iterate":
            return False

        delta_fixed_ref = abs(
            self.relax_energies[irelax, 0] - self.relax_energies[irelax - 1, 0]
        )
        delta_relaxed_ref = abs(
            self.relax_energies[irelax, 1] - self.relax_energies[irelax - 1, 1]
        )
        delta = abs(self.relax_energies[irelax, 1] - self.relax_energies[irelax, 0])

        if all(e < self.relax_tol for e in [delta_fixed_ref, delta_relaxed_ref, delta]):
            return True
        else:
            return False

    @abstractmethod
    def solve_dsrg(self): ...

    @abstractmethod
    def do_reference_relaxation(self): ...

    @abstractmethod
    def get_integrals(self): ...