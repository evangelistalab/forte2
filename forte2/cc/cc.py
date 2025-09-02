from dataclasses import dataclass, field
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np

from forte2.helpers.diis import DIIS
from forte2.helpers import logger
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
    build_nvirt: int = 2
    build_hbar_nvirt: int = 3

    ### CC intermediates
    intermediates: dict = field(default_factory=dict)
    BT1: dict = field(default_factory=dict)

    ### Jacobi parameters
    maxiter: int = 100
    econv: float = 1e-10
    rconv: float = 1e-8
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
        self.method = self._cc_type().upper()

    def __call__(self, scf):
        self.ints = SONormalOrderedIntegrals(scf, 
                                             frozen=self.frozen, 
                                             virtual=self.virtual,
                                             build_nvirt=self.build_nvirt) 
        self.no, self.nu = self.ints['ov'].shape
        return self

    def run(self):
        self._cc_print_banner()

        diis = DIIS(
            diis_start=self.diis_start,
            diis_nvec=self.diis_nvec,
            diis_min=self.diis_min,
            do_diis=self.do_diis,
        )

        if self.first_run:
            self._build_energy_denominators()
            self._build_initial_guess()
            self.first_run = False

        # Compute initial guess energy
        self._build_energy()
        logger.log(f"Initial CC Correlation Energy: {self.E} hartree", self.log_level)

        self.converged = False
        E_old = 0.

        logger.log(
            ("=" * 64)
            + f"\nIter                 E             ΔE        |r|\n"
            + ("-" * 64),
            self.log_level,
        )

        for it in range(self.maxiter):

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
                f"{it:4d}  {self.E:18.12f}  {delta_e:18.12f}  {self.resnorm:12.9f}",
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

            logger.log(f"Reference Energy: {self.ints.E} hartree", self.log_level)
            logger.log(f"Correlation Energy: {self.E} hartree", self.log_level)
            logger.log(f"Total Coupled-Cluster Energy: {self.ints.E + self.E} hartree\n", self.log_level)
        else:
            logger.log("\nCoupled-cluster calculation did not converge.\n", self.log_level)

        # T1-transform DF tensors at the end to make them consistent with T
        self._t1_transformation()

        self.executed = True

        return self
    
    def _cc_print_banner(self):
        logger.log(("-" * 64), self.log_level)
        logger.log(f"Coupled-cluster calculation: {self.method}", self.log_level)
        logger.log(f"Number of occupied spinorbitals: {self.no}", self.log_level)
        logger.log(f"Number of unoccupied spinorbitals: {self.nu}", self.log_level)
        logger.log(f"Number of frozen core spinorbitals: {self.frozen}", self.log_level)
        logger.log(f"Number of frozen virtual spinorbitals: {self.virtual}", self.log_level)
        logger.log(f"Energy convergence: {self.econv}", self.log_level)
        logger.log(f"Amplitude convergence: {self.rconv}", self.log_level)
        if self.do_diis:
            logger.log(f"DIIS subspace size: {self.diis_nvec}", self.log_level)
            logger.log(f"DIIS start: {self.diis_start}", self.log_level)
            logger.log(f"DIIS min: {self.diis_min}", self.log_level)
        logger.log(("-" * 64), self.log_level)

    def _cc_type(self):
        return type(self).__name__.upper()
    
    def _t1_transformation(self):
        t1 = self.T[0]

        # Guard against cases where T1 is absent (e.g., CCD)
        if len(t1.shape) != 2:
            raise ValueError('Dimension of T1 for DF transformation is wrong.')
        
        # T1-transform DF tensor vectors
        self.BT1['ov'] = self.ints.B['ov'].copy()
        self.BT1['oo'] = self.ints.B['oo'].copy() + np.einsum("xme,ei->xmi", self.BT1['ov'], t1, optimize=True)
        self.BT1['vv'] = self.ints.B['vv'].copy() - np.einsum("xme,am->xae", self.BT1['ov'], t1, optimize=True)
        self.BT1['vo'] = (
                self.ints.B['vo'].copy()
                - np.einsum("xmi,am->xai", self.BT1['oo'], t1, optimize=True)
                + np.einsum("xae,ei->xai", self.BT1['vv'], t1, optimize=True)
                + np.einsum("xme,ei,am->xai", self.BT1['ov'], t1, t1, optimize=True)
        )

    @abstractmethod
    def _build_energy_denominators(self): ...
    
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

    @abstractmethod
    def _build_hbar(self): ... 
