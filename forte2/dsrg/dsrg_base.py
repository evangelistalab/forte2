from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np

from forte2.base_classes import SystemMixin, MOsMixin, MOSpaceMixin, ActiveSpaceSolver
from forte2.helpers import logger
from forte2.orbitals import Semicanonicalizer
from forte2.ci.ci_utils import pretty_print_ci_summary


@dataclass
class DSRGBase(SystemMixin, MOsMixin, MOSpaceMixin, ABC):
    """Base class for DSRG methods."""

    # ci_solver: CISolver
    flow_param: float = 0.5

    # Reference relaxation options
    relax_reference: int | str | bool = False
    relax_maxiter: int = 10
    relax_tol: float = 1e-6

    # options to freeze orbitals
    frozen_core_orbitals: int | list[int] = None
    frozen_virtual_orbitals: int | list[int] = None

    # Non-init attributes
    executed: bool = field(init=False, default=False)
    converged: bool = field(init=False, default=False)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        assert isinstance(
            self.parent_method, ActiveSpaceSolver
        ), "Parent method must be an ActiveSpaceSolver."
        # This is to ensure that the CI vectors are converged after
        # the basis is changed to semicanonical orbitals.
        # We could handle it here, but it's cleaner to enforce it at the parent method level.
        assert (
            self.parent_method.final_orbital.lower() == "semicanonical"
        ), "The final_orbital of the parent method must be 'semicanonical' for DSRG methods."
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
        self.relax_eigvals_history = []

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        self.two_component = self.system.two_component

        MOSpaceMixin.copy_from_upstream(self, self.parent_method)

        # update the MOSpace object if frozen orbitals are specified
        if (
            self.frozen_core_orbitals is not None
            or self.frozen_virtual_orbitals is not None
        ):
            self.mo_space = self.mo_space.update_frozen_orbitals(
                frozen_core_orbitals=self.frozen_core_orbitals,
                frozen_virtual_orbitals=self.frozen_virtual_orbitals,
            )

        self.ncorr = self.mo_space.corr.stop - self.mo_space.corr.start
        self.ncore = self.mo_space.core_corr.stop - self.mo_space.core_corr.start
        self.nact = self.mo_space.actv_corr.stop - self.mo_space.actv_corr.start
        self.nvirt = self.mo_space.virt_corr.stop - self.mo_space.virt_corr.start
        self.nhole = self.ncore + self.nact
        self.npart = self.nact + self.nvirt
        self.frozen_core = self.mo_space.frozen_core
        self.corr = self.mo_space.corr
        self.actv = self.mo_space.actv_corr
        self.core = self.mo_space.core_corr
        self.virt = self.mo_space.virt_corr
        self.hole = slice(0, self.nhole)
        self.part = slice(self.ncore, self.ncorr)
        self.ha = self.actv
        self.pa = slice(0, self.nact)
        self.hc = self.core
        self.pv = slice(self.nact, self.nact + self.nvirt)

        MOsMixin.copy_from_upstream(self, self.parent_method)
        perm = self.mo_space.orig_to_contig
        self._C = self.C[0][:, perm].copy()

        # TODO: this interface should be homogenized
        if hasattr(self.parent_method, "ci_solver"):
            # parent method is RelMCOptimizer
            self.ci_solver = self.parent_method.ci_solver
        else:
            # parent method is RelCISolver
            self.ci_solver = self.parent_method

        self.E_core_orig = self.ci_solver.sub_solvers[0].ints.E
        self.H_orig = self.ci_solver.sub_solvers[0].ints.H.copy()
        self.V_orig = self.ci_solver.sub_solvers[0].ints.V.copy()

        self.semicanonicalizer = Semicanonicalizer(
            system=self.system,
            mo_space=self.mo_space,
            mix_active=False,
            # do not mix correlated core and frozen core orbitals after MCSCF
            mix_inactive=False,
        )

        self.fock_builder = self.system.fock_builder
        self.ints, self.cumulants = self.get_integrals()
        self.hbar = dict()

    def run(self):
        self._startup()
        form_hbar = self.nrelax > 0

        self.E_dsrg = self.solve_dsrg(form_hbar)
        if abs(self.E_dsrg.imag) > 1e-10:
            logger.log_warning(
                f"DSRG energy has a significant imaginary component: {self.E_dsrg.imag}"
            )

        self.relax_energies[0, 0] = self.E_dsrg.real
        # self.ints["E"] is <Psi_current| bare H |Psi_current>
        self.relax_energies[0, 2] = self.ints["E"].real
        self.E = self.E_dsrg

        width = 88
        for irelax in range(self.nrelax):
            if irelax == 0:
                logger.log_info1("\n DSRG reference relaxation glossary")
                logger.log_info1(" -E0      : <Psi_n     | bare H | Psi_n    >")
                logger.log_info1(" -Edsrg   : <Psi_n     | Hbar   | Psi_n    >")
                logger.log_info1(" -Erelaxed: <Psi_{n+1} | Hbar   | Psi_{n+1}>")
                logger.log_info1("=" * width)
                logger.log_info1("DSRG Reference Relaxation Summary".center(width))
                logger.log_info1("=" * width)
                logger.log_info1(
                    f"{'Iteration':>10} {'E0 (a.u.)':>25} {'Edsrg (a.u.)':>25} {'Erelaxed (a.u.)':>25}"
                )
                logger.log_info1("-" * width)
            self.relax_energies[irelax, 0] = self.E_dsrg.real
            self.relax_energies[irelax, 2] = self.ints["E"].real
            self.E_relaxed_ref = self.do_reference_relaxation()
            self.relax_eigvals_history.append(self.relax_eigvals)
            self.relax_energies[irelax, 1] = self.E_relaxed_ref.real

            logger.log_info1(
                f"{irelax:>10d} {self.relax_energies[irelax,2]:>25.12f} {self.relax_energies[irelax,0]:>25.12f} {self.relax_energies[irelax,1]:>25.12f}"
            )

            # "once": DSRG -> relax -> done
            if self.relax_reference == "once":
                self.converged = True
                break

            self.converged = self.test_relaxation_convergence(irelax)
            if self.converged:
                break

            self.ints, self.cumulants = self.get_integrals()
            self.E_dsrg = self.solve_dsrg(form_hbar=form_hbar)
            self.E = self.E_dsrg
        else:
            logger.log_warning(
                f"DSRG reference relaxation did not converge in {self.nrelax} iterations."
            )
        logger.log_info1("=" * width)
        logger.log_info1("\nFinal DSRG energies (a.u.):")
        logger.log_info1(f"  E0       : {self.ints['E'].real:.12f}")
        logger.log_info1(f"  Edsrg    : {self.E_dsrg.real:.12f}")
        if self.nrelax > 0:
            logger.log_info1(f"  Erelaxed : {self.E_relaxed_ref.real:.12f}")
            if len(self.relax_eigvals) > 1:
                logger.log_info1("")
                pretty_print_ci_summary(
                    self.ci_solver.sa_info, self.ci_solver.evals_per_solver
                )
        self.relax_eigvals_history = np.array(self.relax_eigvals_history)
        self.executed = True
        return self

    def test_relaxation_convergence(self, irelax):
        if irelax == 0:
            return False

        if irelax == self.nrelax - 1:
            return True

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
