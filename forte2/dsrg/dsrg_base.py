from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from forte2.base_classes import SystemMixin, MOsMixin, MOSpaceMixin, ActiveSpaceSolver
from forte2.helpers import logger
from forte2.jkbuilder import FockBuilder


@dataclass
class DSRGBase(SystemMixin, MOsMixin, MOSpaceMixin, ABC):
    """
    Base class for DSRG methods.

    Parameters
    ----------
    flow_param : float, optional, default=0.5
        The flow parameter (in atomic units) that controls the renormalization.
    relax_reference : int | str | bool, optional, default=False
        Relax the CI reference in response to dynamical correlation.
        If an integer is given, it specifies the maximum number of relaxation iterations.
        If a string is given, it must be one of 'once', 'twice', or 'iterate':
            'once' : diagonalize the CI Hamiltonian once after computing the DSRG energy
            'twice': after the first diagonalization, recompute the DSRG energy
            'iterate': keep relaxing until convergence or reaching relax_maxiter.
        If a boolean is given, True is equivalent to relax_maxiter and False means no relaxation.
    relax_maxiter : int, optional, default=10
        The maximum number of reference relaxation iterations.
    relax_tol : float, optional, default=1e-6
        The convergence tolerance for reference relaxation (in Hartree).

    References
    ----------
    .. [1] F. A. Evangelista, "A driven similarity renormalization group approach to quantum many-body problems",
           J. Chem. Phys. 2014, 141, 054109.
    .. [2] C. Li and F. A. Evangelista, "Multireference driven similarity renormalization group: A second-order perturbative analysis",
           J. Chem. Theory Comput. 2015, 11, 2097-2108.
    .. [3] K. P. Hannon, C. Li, and F. A. Evangelista, "An integral-factorized implementation of the driven similarity renormalization group second-order multireference perturbation theory",
              J. Chem. Phys. 2016, 144, 204111.
    .. [4] C. Li and F. A. Evangelista, "Driven similarity renormalization group for excited states: A state-averaged perturbation theory",
           J. Chem. Phys. 2018, 148, 124106.
    """

    # ci_solver: CISolver
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
        else:
            logger.log_warning(
                "Reference relaxation options not recognized, no relaxation will be performed."
            )
            self.nrelax = 0
        # [Edsrg(fixed_reference), Edsrg(relaxed_reference), Eref]
        self.relax_energies = np.zeros((self.nrelax + 1, 3))

    def _startup(self):
        assert isinstance(self.parent_method, ActiveSpaceSolver)
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        self.two_component = self.system.two_component

        MOSpaceMixin.copy_from_upstream(self, self.parent_method)
        perm = self.mo_space.orig_to_contig
        self.ncorr = self.mo_space.corr.stop - self.mo_space.corr.start
        self.ncore = self.mo_space.core.stop - self.mo_space.core.start
        self.nact = self.mo_space.actv.stop - self.mo_space.actv.start
        self.nvirt = self.mo_space.virt.stop - self.mo_space.virt.start
        self.nhole = self.ncore + self.nact
        self.npart = self.nact + self.nvirt
        self.actv = self.mo_space.actv
        self.core = self.mo_space.core
        self.virt = self.mo_space.virt
        self.hole = slice(0, self.nhole)
        self.part = slice(self.ncore, self.ncorr)
        self.ha = self.actv
        self.pa = slice(0, self.nact)
        self.hc = self.core
        self.pv = slice(self.nact, self.nact + self.nvirt)

        MOsMixin.copy_from_upstream(self, self.parent_method)
        self._C = self.C[0][:, perm].copy()

        # only initialize the a new CI solver if reference relaxation is requested
        # initialized in do_reference_relaxation()
        self.ci_solver = None

        self.fock_builder = FockBuilder(system=self.system, use_aux_corr=True)

        self.ints, self.cumulants = self.get_integrals()

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
