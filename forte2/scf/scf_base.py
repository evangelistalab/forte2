from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

import numpy as np
from forte2.system import System, ModelSystem, BasisInfo
from forte2.jkbuilder import FockBuilder
from forte2.base_classes.mixins import MOsMixin, SystemMixin
from forte2.helpers import logger, DIIS


@dataclass
class SCFBase(ABC, SystemMixin, MOsMixin):
    """
    Abstract base class for SCF calculations.

    Parameters
    ----------
    charge : int
        Charge of the system.
    do_diis : bool, optional, default=True
        Whether to perform DIIS acceleration.
    diis_start : int, optional, default=4
        Which iteration to start collecting DIIS error vectors.
    diis_nvec : int, optional, default=8
        How many DIIS error vectors to keep.
    diis_min : int, optional, default=3
        Minimum number of DIIS vectors to perform extrapolation.
    econv : float, optional, default=1e-9
        Energy convergence threshold.
    dconv : float, optional, default=1e-6
        RMS density change convergence threshold.
    maxiter : int, optional, default=100
        Maximum iteration for SCF.
    guess_type : str, optional, default="minao"
        Initial guess type for the SCF calculation. Can be "minao" or "hcore".
    level_shift : float, optional
        Level shift for the SCF calculation. If None, no level shift is applied.
    level_shift_thresh : float, optional, default=1e-5
        If energy change is below this threshold, level shift is turned off.
    die_if_not_converged : bool, optional, default=True
        Whether to raise an error if the SCF calculation does not converge.

    Attributes
    ----------
    C : list[NDArray]
        The MO coefficients.
    D : list[NDArray]
        The density matrices.
    E : float
        The total energy of the system.
    F : list[NDArray]
        The Fock matrices.
    eps : list[NDArray]
        The orbital energies.

    Raises
    ------
    RuntimeError
        If the SCF calculation does not converge within the maximum number of iterations.
    """

    charge: int
    do_diis: bool = True
    diis_start: int = 1
    diis_nvec: int = 8
    diis_min: int = 2
    econv: float = 1e-9
    dconv: float = 1e-6
    maxiter: int = 100
    guess_type: str = "minao"
    level_shift: float = None
    level_shift_thresh: float = 1e-5
    die_if_not_converged: bool = True

    executed: bool = field(default=False, init=False)
    converged: bool = field(default=False, init=False)

    def __call__(self, system):
        assert isinstance(
            system, (System, ModelSystem)
        ), "System must be an instance of forte2.System"
        self.system = system
        self.method = self._scf_type().upper()
        self.nel = self.system.Zsum - self.charge
        assert self.nel >= 0, "Number of electrons must be non-negative."

        if self.method != "GHF" and self.system.x2c_type == "so":
            raise ValueError(
                "SO-X2C is only available for GHF. Use SF-X2C for RHF/UHF."
            )

        self.C = None
        self.Xorth = self.system.get_Xorth()
        if self.level_shift is not None:
            if isinstance(self.level_shift, (int, float)) and self.level_shift < 0.0:
                raise ValueError("level_shift must be non-negative.")
            if isinstance(self.level_shift, tuple) and self.method != "UHF":
                raise ValueError("Tuple level_shift is only valid for UHF.")
            if isinstance(self.level_shift, float) and self.method == "UHF":
                self.level_shift = (self.level_shift, self.level_shift)
            if isinstance(self.level_shift, tuple) and len(self.level_shift) != 2:
                raise ValueError("Tuple level_shift must have length 2 for UHF.")
        return self

    def _eigh(self, F):
        Ftilde = self.Xorth.T @ F @ self.Xorth
        e, c = np.linalg.eigh(Ftilde)
        return e, self.Xorth @ c

    def _scf_type(self):
        return type(self).__name__.upper()

    def run(self):
        """
        Run the SCF calculation.

        Returns
        -------
            self : SCFBase
                The SCF object.
        """
        start = time.monotonic()

        diis = DIIS(
            diis_start=self.diis_start,
            diis_nvec=self.diis_nvec,
            diis_min=self.diis_min,
            do_diis=self.do_diis,
        )
        Vnn = self._get_nuclear_repulsion()
        S = self._get_overlap()
        H = self._get_hcore()
        fock_builder = FockBuilder(self.system)

        self.nbf = self.system.nbf
        self.naux = self.system.naux
        self.nmo = self.system.nmo

        self.basis_info = BasisInfo(self.system, self.system.basis)

        logger.log_info1(f"Number of electrons: {self.nel}")
        if self._scf_type() != "GHF":  # not good quantum numbers for GHF
            logger.log_info1(f"Number of alpha electrons: {self.na}")
            logger.log_info1(f"Number of beta electrons: {self.nb}")
            logger.log_info1(f"Ms: {self.ms}")
        logger.log_info1(f"Total charge: {self.charge}")
        logger.log_info1(f"Number of basis functions: {self.nbf}")
        logger.log_info1(f"Number of orthogonalized basis functions: {self.nmo}")
        logger.log_info1(f"Number of auxiliary basis functions: {self.naux}")
        logger.log_info1(f"Energy convergence criterion: {self.econv:e}")
        logger.log_info1(f"Density convergence criterion: {self.dconv:e}")
        logger.log_info1(f"DIIS acceleration: {diis.do_diis}")
        logger.log_info1(f"\n==> {self.method} SCF ROUTINE <==")
        self.iter = 0
        if self.C is None:
            self.C = self._initial_guess(H, guess_type=self.guess_type)
        self.D = self._build_density_matrix()
        F, F_canon = self._build_fock(H, fock_builder, S)
        self.F = F_canon
        self.E = Vnn + self._energy(H, F)

        Eold = self.E
        Dold = self.D
        self.iter += 1

        width = 81
        logger.log_info1("=" * width)
        logger.log_info1(
            f"{'Iter':>4s} {'Energy':>20s} {'ΔE':>12} {'||ΔD||':>12} {'||AO grad||':>12} {'<S^2>':>10} {'DIIS':>5s}"
        )
        logger.log_info1("-" * width)
        for iter in range(self.maxiter):
            # 1. Get the extrapolated Fock matrix
            AO_grad = self._build_ao_grad(S, F_canon)
            F_canon = self._diis_update(diis, F_canon, AO_grad)
            F_canon = self._apply_level_shift(F_canon, S)
            # 2. Diagonalize the extrapolated Fock
            self.eps, self.C = self._diagonalize_fock(F_canon)
            # 3. Build new density matrix
            self.D = self._build_density_matrix()
            # 4. Build the (non-extrapolated) Fock matrix
            # (there is a slot for canonicalized F to accommodate ROHF and CUHF methods - admittedly weird for RHF/UHF)
            F, F_canon = self._build_fock(H, fock_builder, S)
            self.F = F_canon
            # 5. Compute new HF energy from the non-extrapolated Fock matrix
            self.E = Vnn + self._energy(H, F)

            # check convergence parameters
            deltaE = self.E - Eold
            if np.abs(deltaE) < self.level_shift_thresh:
                self.level_shift = None
            deltaD = sum([np.linalg.norm(d - dold) for d, dold in zip(self.D, Dold)])
            self.S2 = self._spin(S)

            # print iteration
            logger.log_info1(
                f"{iter+1:4d} {self.E:20.12f} {deltaE:12.4e} {deltaD:12.4e} {np.linalg.norm(AO_grad):12.4e} {self.S2:10.5f} {diis.status:>5s}"
            )

            if np.abs(deltaE) < self.econv and deltaD < self.dconv:
                logger.log_info1("=" * width)
                logger.log_info1(f"{self.method} iterations converged\n")
                # perform final iteration
                self.eps, self.C = self._diagonalize_fock(F_canon)
                self.D = self._build_density_matrix()
                F, F_canon = self._build_fock(H, fock_builder, S)
                self.F = F_canon
                self.E = Vnn + self._energy(H, F)
                logger.log_info1(f"Final {self.method} Energy: {self.E:20.12f}")
                self.converged = True
                break

            # reset old parameters
            Eold = self.E
            Dold = self.D
            self.iter += 1
        else:
            logger.log_info1("=" * width)
            logger.log_info1(f"{self.method} iterations did not converge")
            if self.die_if_not_converged:
                raise RuntimeError(
                    f"{self.method} did not converge in {self.maxiter} iterations."
                )
            else:
                logger.log_warning(
                    f"{self.method} did not converge in {self.maxiter} iterations."
                )

        end = time.monotonic()
        logger.log_info1(f"{self.method} time: {end - start:.2f} seconds")

        self._post_process()

        self.executed = True
        return self

    def _get_hcore(self):
        return self.system.ints_hcore()

    def _get_overlap(self):
        return self.system.ints_overlap()

    def _get_nuclear_repulsion(self):
        return self.system.nuclear_repulsion

    def _post_process(self):
        self._get_occupation()
        self._assign_orbital_symmetries()
        self._print_orbital_energies()
        self._print_ao_composition()

    @abstractmethod
    def _build_fock(self, H, fock_builder, S): ...

    @abstractmethod
    def _build_density_matrix(self): ...

    @abstractmethod
    def _initial_guess(self, H, guess_type="minao"): ...

    @abstractmethod
    def _build_ao_grad(self, S, F): ...

    @abstractmethod
    def _diagonalize_fock(self, F): ...

    @abstractmethod
    def _spin(self, S): ...

    @abstractmethod
    def _energy(self, H, F): ...

    @abstractmethod
    def _diis_update(self, diis, F, AO_grad): ...

    @abstractmethod
    def _get_occupation(self): ...

    @abstractmethod
    def _print_orbital_energies(self): ...

    @abstractmethod
    def _assign_orbital_symmetries(self): ...

    @abstractmethod
    def _apply_level_shift(self, F, S): ...

    @abstractmethod
    def _print_ao_composition(self): ...
