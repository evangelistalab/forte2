from dataclasses import dataclass, field
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np

from forte2.helpers.diis import DIIS
from forte2.helpers import logger
# from forte2.helpers.jacobi import JacobiSolver
from forte2.jkbuilder import SONormalOrderedIntegrals


@dataclass
class _SRCCBase(ABC):
    """
    A general single-reference coupled-cluster (CC) solver class to compute the ground
    state (or lowest state of a particular symmetry). 
    Although possible, is not recommended to instantiate this class directly.
    Consider using the `SRCC` class instead.

    Parameters
    ----------
    log_level : int, optional
        The logging level for the CC solver. Defaults to the global logger's verbosity level.
    maxiter : int, optional, default=100
        The maximum number of iterations for the Davidson-Liu solver.
    econv : float, optional, default=1e-10
        The energy convergence threshold for the solver.
    rconv : float, optional, default=1e-5
        The residual convergence threshold for the solver.
    energy_shift : float, optional, default=None
        An energy shift to find roots around. If None, no shift is applied.

    Attributes
    ----------
    E : float
        The CC correlation energy.
    T : NDArray
        The flattened array of cluster amplitudes

    """

    scf: object | None = None
    frozen: int = 0
    virtual: int = 0
    log_level: int = field(default=logger.get_verbosity_level())

    ### CC intermediates
    intermediates: dict = field(default_factory=dict)
    BT1: dict = field(default_factory=dict)

    ### Jacobi parameters
    maxiter: int = 100
    econv: float = 1e-10
    rconv: float = 1e-5
    energy_shift: float = 0.

    ### DIIS parameters
    diis_start: int = 4
    diis_nvec: int = 6
    diis_min: int = 3
    do_diis: bool = True

    ### Non-init attributes
    first_run: bool = field(default=True, init=False)
    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.scf is not None:
            self(scf=self.scf)

    def __call__(self, scf):
        self.ints = SONormalOrderedIntegrals(scf, 
                                             frozen=self.frozen, 
                                             virtual=self.virtual,
                                             build_nvirt=2) 
        self.no, self.nu = self.ints['ov'].shape
        return self

    def run(self):
        # if not self.executed:
        #     self._cc_solver_startup()

        diis = DIIS(
            diis_start=self.diis_start,
            diis_nvec=self.diis_nvec,
            diis_min=self.diis_min,
            do_diis=self.do_diis,
        )

        if self.first_run:
            self._build_initial_guess()
            self._build_energy()
            self.first_run = False

        logger.log(f"Initial CC Correlation Energy: {self.E} hartree", self.log_level)

        # 5. Run Jacobi
        # T, E = self.solver.solve(T, self.ints)
        self.converged = False
        E_old = 0.

        logger.log(
            ("=" * 64)
            + f"\nIter                 E             ΔE        |r|\n"
            + ("-" * 64),
            self.log_level,
        )

        for iter in range(self.maxiter):

            # 1. Compute R_K = < K | (H e^T)_C | 0 >
            self._build_residual()
            # 2. Update T <- T + R_K / D_K
            self._build_update()
            # 3. Compute correlation energy E = < 0 | (H e^T)_C | 0 >
            self._build_energy()
            # 4. Compute convergence parameters
            delta_e = self.E - E_old
            # print the iteration
            logger.log(
                f"{iter:4d}  {self.E:18.12f}  {delta_e:18.12f}  {self.resnorm:12.9f}",
                self.log_level,
            )
            # check exit conditions
            if abs(delta_e) < self.econv and self.resnorm < self.rconv:
                self.converged = True
                break
            E_old = self.E
            # 5. DIIS update
            self._diis_update(diis)

        logger.log(
            ("=" * 64),
            self.log_level,
        )
        if self.converged:
            logger.log("\nCoupled-cluster calculation converged.\n", self.log_level)
        else:
            logger.log("\nCoupled-cluster calculation did not converge.\n", self.log_level)

        self.executed = True

        return self
    
    @abstractmethod
    def _build_residual(self): ...   

    @abstractmethod
    def _build_update(self): ...  

    @abstractmethod
    def _build_energy(self): ... 

    @abstractmethod
    def _build_initial_guess(self): ...

    @abstractmethod
    def _diis_update(self, diis): ... 
