from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

import numpy as np
import scipy as sp

from forte2.system.basis_utils import BasisInfo
from forte2.system import System, ModelSystem, compute_orthonormal_transformation
from forte2.jkbuilder import FockBuilder
from forte2.base_classes.mixins import MOsMixin, SystemMixin
from forte2.helpers import logger, DIIS
from forte2.symmetry import assign_mo_symmetries, real_sph_to_j_adapted
from .scf_utils import (
    minao_initial_guess,
    core_initial_guess,
    guess_mix,
    alpha_beta_mix,
    break_complex_conjugation_symmetry,
)


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
    #: Initial guess algorithm. Can be "minao" or "hcore".
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


@dataclass
class RHF(SCFBase):
    """
    A class that runs restricted Hartree-Fock calculations.
    """

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _parse_state(self):
        assert self.nel % 2 == 0, "RHF requires an even number of electrons."
        self.ms = 0
        self.na = self.nb = self.nel // 2

    def _build_fock(self, H, fock_builder, S):
        J = fock_builder.build_J(self.D)[0]
        K = fock_builder.build_K([self.C[0][:, : self.na]])[0]
        F = H + 2.0 * J - K
        return [F], [F]

    def _build_density_matrix(self):
        D = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        return [D]

    def _build_total_density_matrix(self):
        # returns the total density matrix (Daa + Dbb)
        return 2 * self._build_density_matrix()[0]

    def _initial_guess(self, H, guess_type="minao"):
        match guess_type:
            case "minao":
                C = minao_initial_guess(self.system, H)
            case "hcore":
                C = core_initial_guess(self.system, H)
            case _:
                raise RuntimeError(f"Unknown initial guess type: {guess_type}")

        return [C]

    def _build_ao_grad(self, S, F):
        ao_grad = F[0] @ self.D[0] @ S - S @ self.D[0] @ F[0]
        ao_grad = self.Xorth.T @ ao_grad @ self.Xorth
        return ao_grad

    def _energy(self, H, F):
        return np.sum(self.D[0] * (H + F[0]))

    def _diagonalize_fock(self, F):
        eps, C = self._eigh(F[0])
        return [eps], [C]

    def _spin(self, S):
        return self.ms * (self.ms + 1)

    def _diis_update(self, diis, F, AO_grad):
        return [diis.update(F[0], AO_grad)]

    def _apply_level_shift(self, F, S):
        if self.level_shift is None or self.level_shift < 1e-4:
            return F
        D_vir = S - S @ self.D[0] @ S

        return [F[0] + self.level_shift * D_vir]

    def _get_occupation(self):
        self.ndocc = self.na
        self.nuocc = self.nmo - self.ndocc

    def _print_orbital_energies(self):
        ndocc = self.na
        nuocc = self.nmo - ndocc
        orb_per_row = 5
        logger.log_info1("---------------------")
        logger.log_info1("Orbital Energies [Eh]")
        logger.log_info1("---------------------")
        logger.log_info1("Doubly Occupied:")
        string = ""
        for i in range(ndocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = ndocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
            )
        logger.log_info1(string)

    def _post_process(self):
        super()._post_process()
        self._print_ao_composition()

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        logger.log_info1("\nAO Composition of MOs (HOMO-5 to HOMO):")
        basis_info.print_ao_composition(
            self.C[0], list(range(max(self.na - 5, 0), self.na))
        )
        logger.log_info1("\nAO Composition of MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.na, min(self.na + 5, self.nmo)))
        )

    def _assign_orbital_symmetries(self):
        S = self._get_overlap()
        self.irrep_labels, self.irrep_indices = assign_mo_symmetries(
            self.system, S, self.C[0]
        )


@dataclass
class UHF(SCFBase):
    """
    A class that runs unrestricted Hartree-Fock calculations.

    Parameters
    ----------
    ms : float
        Spin projection. Must be a multiple of 0.5.
    guess_mix : bool, optional, default=False
        If True, will mix the HOMO and LUMO orbitals to try to break alpha-beta degeneracy if ms is 0.0.
    """

    ms: float = None
    guess_mix: bool = False  # only used if ms == 0

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _parse_state(self):
        if self.ms is None:
            raise ValueError(
                f"Spin projection (ms) must be set for {self._scf_type()}."
            )
        assert np.isclose(
            int(round(self.ms * 2)), self.ms * 2
        ), "ms must be a multiple of 0.5."
        self.twicems = int(round(self.ms * 2))
        if self.nel % 2 != self.twicems % 2:
            raise ValueError(f"{self.nel} electrons is incompatible with ms={self.ms}!")
        self.na = int(round(self.nel + self.twicems) / 2)
        self.nb = int(round(self.nel - self.twicems) / 2)
        assert (
            self.nel == self.na + self.nb
        ), f"Number of electrons {self.nel} does not match na + nb = {self.na} + {self.nb}."
        assert (
            self.na >= 0 and self.nb >= 0
        ), f"{self._scf_type} requires non-negative number of alpha and beta electrons."

    def _build_fock(self, H, fock_builder, S):
        Ja, Jb = fock_builder.build_J(self.D)
        K = fock_builder.build_K([self.C[0][:, : self.na], self.C[1][:, : self.nb]])
        F = [H + Ja + Jb - k for k in K]

        F_canon = F

        return F, F_canon

    def _build_density_matrix(self):
        D_a = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        D_b = np.einsum("mi,ni->mn", self.C[1][:, : self.nb], self.C[1][:, : self.nb])
        return D_a, D_b

    def _build_total_density_matrix(self):
        D_a, D_b = self._build_density_matrix()
        return D_a + D_b

    def _initial_guess(self, H, guess_type="minao"):
        C = RHF._initial_guess(self, H, guess_type=guess_type)[0]

        if self.twicems == 0 and self.guess_mix:
            return guess_mix(C, self.nel // 2 - 1)

        return [C, C]

    def _build_ao_grad(self, S, F):
        AO_grad = np.hstack(
            [
                (self.Xorth.T @ (f @ d @ S - S @ d @ f) @ self.Xorth).flatten()
                for d, f in zip(self.D, F)
            ]
        )
        return AO_grad

    def _diagonalize_fock(self, F):
        eps_a, C_a = self._eigh(F[0])
        eps_b, C_b = self._eigh(F[1])
        return [eps_a, eps_b], [C_a, C_b]

    def _spin(self, S):
        # alpha-beta orbital overlap matrix
        # S_ij = < psi_i | psi_j >, i,j=occ
        #      = \sum_{uv} c_ui^* c_vj <u|v>
        S_ij = np.einsum(
            "ui,uv,vj->ij",
            self.C[0][:, : self.na].conj(),
            S,
            self.C[1][:, : self.nb],
            optimize=True,
        )
        # Spin contamination: <s^2> - <s^2>_exact = N_b - \sum_{ij} |S_ij|^2
        ds2 = self.nb - np.einsum("ij,ij->", S_ij.conj(), S_ij)
        # <S^2> value
        s2 = self.ms * (self.ms + 1) + ds2
        return s2

    def _energy(self, H, F):
        energy = 0.5 * (
            np.einsum("vu,uv->", self.D[0] + self.D[1], H)
            + np.einsum("vu,uv->", self.D[0], F[0])
            + np.einsum("vu,uv->", self.D[1], F[1])
        )
        return energy

    def _diis_update(self, diis, F, AO_grad):
        F_flat = diis.update(np.hstack([f.flatten() for f in F]), AO_grad)
        F = [
            F_flat[: self.nbf**2].reshape(self.nbf, self.nbf),
            F_flat[self.nbf**2 :].reshape(self.nbf, self.nbf),
        ]
        return F

    def _apply_level_shift(self, F, S):
        if self.level_shift is None or all(ls < 1e-4 for ls in self.level_shift):
            return F
        D_vir = [S - S @ d @ S for d in self.D]

        return [f + ls * d for ls, f, d in zip(self.level_shift, F, D_vir)]

    def _get_occupation(self):
        self.aocc = self.na
        self.auocc = self.nmo - self.aocc
        self.bocc = self.nb
        self.buocc = self.nmo - self.bocc

    def _print_orbital_energies(self):
        naocc = self.na
        naucc = self.nmo - naocc
        nbocc = self.nb
        nbucc = self.nmo - nbocc
        orb_per_row = 5
        logger.log_info1("---------------------")
        logger.log_info1("Orbital Energies [Eh]")
        logger.log_info1("---------------------")
        logger.log_info1("Alpha Occupied:")
        string = ""
        for i in range(naocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[0][i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nAlpha Virtual:")
        string = ""
        for i in range(naucc):
            idx = naocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[0][idx]}) {self.eps[0][idx]:<12.6f} "
            )
        logger.log_info1(string)

        logger.log_info1("\nBeta Occupied:")
        string = ""
        for i in range(nbocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[1][i]}) {self.eps[1][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nBeta Virtual:")
        string = ""
        for i in range(nbucc):
            idx = nbocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[1][idx]}) {self.eps[1][idx]:<12.6f} "
            )
        logger.log_info1(string)

    def _assign_orbital_symmetries(self):
        S = self._get_overlap()
        irrep_labels_alpha, irrep_indices_alpha = assign_mo_symmetries(
            self.system, S, self.C[0]
        )
        irrep_labels_beta, irrep_indices_beta = assign_mo_symmetries(
            self.system, S, self.C[1]
        )
        self.irrep_labels = [irrep_labels_alpha, irrep_labels_beta]
        self.irrep_indices = [irrep_indices_alpha, irrep_indices_beta]

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        logger.log_info1("\nAO Composition of Alpha MOs (HOMO-5 to HOMO):")
        basis_info.print_ao_composition(
            self.C[0], list(range(max(self.na - 5, 0), self.na))
        )
        logger.log_info1("\nAO Composition of Alpha MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.na, min(self.na + 5, self.nmo)))
        )
        logger.log_info1("\nAO Composition of Beta MOs (HOMO-5 to HOMO):")
        basis_info.print_ao_composition(
            self.C[1], list(range(max(self.na - 5, 0), self.na))
        )
        logger.log_info1("\nAO Composition of Beta MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[1], list(range(self.na, min(self.na + 5, self.nmo)))
        )


@dataclass
class ROHF(SCFBase):
    """
    A class that runs restricted open-shell Hartree-Fock calculations.

    Parameters
    ----------
    ms : float
        Spin projection. Must be a multiple of 0.5.
    """

    ms: float = None

    _parse_state = UHF._parse_state
    _initial_guess = RHF._initial_guess
    _diagonalize_fock = RHF._diagonalize_fock
    _spin = RHF._spin
    _energy = UHF._energy
    _diis_update = RHF._diis_update
    _build_total_density_matrix = UHF._build_total_density_matrix
    _assign_orbital_symmetries = RHF._assign_orbital_symmetries
    _apply_level_shift = RHF._apply_level_shift

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _build_fock(self, H, fock_builder, S):
        Ja, Jb = fock_builder.build_J(self.D)
        K = fock_builder.build_K([self.C[0][:, : self.na], self.C[0][:, : self.nb]])
        F = [H + Ja + Jb - k for k in K]

        F_canon = self._build_canonical_fock(F, S)

        return F, F_canon

    def _build_canonical_fock(self, F, S):

        # Projection matrices for core (doubly occupied), open (singly occupied, alpha), and virtual (unoccupied) MO spaces
        U_core = np.dot(self.D[1], S)
        U_open = np.dot(self.D[0] - self.D[1], S)
        U_virt = np.eye(self.nbf) - np.dot(self.D[0], S)

        # Closed-shell Fock
        F_cs = 0.5 * (F[0] + F[1])

        def _project(u, v, f):
            return np.einsum("ur,uv,vt->rt", u, f, v, optimize=True)

        # these are scaled by 0.5 to account for fock + fock.T below
        fock = _project(U_core, U_core, F_cs) * 0.5
        fock += _project(U_open, U_open, F_cs) * 0.5
        fock += _project(U_virt, U_virt, F_cs) * 0.5
        # off-diagonal blocks
        fock += _project(U_open, U_core, F[1])
        fock += _project(U_open, U_virt, F[0])
        fock += _project(U_virt, U_core, F_cs)
        fock = fock + fock.conj().T
        return [fock]

    def _build_density_matrix(self):
        D_a = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        D_b = np.einsum("mi,ni->mn", self.C[0][:, : self.nb], self.C[0][:, : self.nb])
        return D_a, D_b

    def _build_ao_grad(self, S, F):
        Deff = 0.5 * (self.D[0] + self.D[1])
        ao_grad = F @ Deff @ S - S @ Deff @ F
        ao_grad = self.Xorth.T @ ao_grad @ self.Xorth
        return ao_grad

    def _get_occupation(self):
        self.ndocc = min(self.na, self.nb)
        self.nsocc = abs(self.na - self.nb)
        self.nuocc = self.nmo - self.ndocc - self.nsocc

    def _print_orbital_energies(self):
        ndocc = min(self.na, self.nb)
        nsocc = abs(self.na - self.nb)
        nuocc = self.nmo - ndocc - nsocc
        orb_per_row = 5
        logger.log_info1("---------------------")
        logger.log_info1("Orbital Energies [Eh]")
        logger.log_info1("---------------------")
        logger.log_info1("Doubly Occupied:")
        string = ""
        for i in range(ndocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        if nsocc > 0:
            logger.log_info1("\nSingly Occupied:")
            string = ""
            for i in range(nsocc):
                idx = ndocc + i
                if i % orb_per_row == 0:
                    string += "\n"
                string += (
                    f"{idx+1:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
                )
            logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = ndocc + nsocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
            )
        logger.log_info1(string)

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        logger.log_info1("\nAO Composition of doubly occupied MOs (HOMO-3 to HOMO):")
        basis_info.print_ao_composition(
            self.C[0], list(range(max(self.ndocc - 3, 0), self.ndocc))
        )
        logger.log_info1("\nAO Composition of singly occupied MOs:")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.ndocc, self.ndocc + self.nsocc))
        )
        logger.log_info1("\nAO Composition of MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[0],
            list(
                range(
                    self.ndocc + self.nsocc, min(self.ndocc + self.nsocc + 5, self.nmo)
                )
            ),
        )


@dataclass
class CUHF(SCFBase):
    """
    A class that runs constrained unrestricted Hartree-Fock calculations.
    Equivalent to ROHF but uses UHF machinery.
    See J. Chem. Phys. 133, 141102 (2010) (10.1063/1.3503173)

    Parameters
    ----------
    ms : float
        Spin projection. Must be a multiple of 0.5.
    guess_mix : bool, optional, default=False
        If True, will mix the HOMO and LUMO orbitals to try to break alpha-beta degeneracy if ms is 0.0.
    """

    ms: float = None
    guess_mix: bool = False  # only used if ms == 0

    _parse_state = UHF._parse_state
    _build_density_matrix = UHF._build_density_matrix
    _initial_guess = UHF._initial_guess
    _build_ao_grad = UHF._build_ao_grad
    _diagonalize_fock = UHF._diagonalize_fock
    _spin = UHF._spin
    _energy = UHF._energy
    _diis_update = UHF._diis_update
    _build_total_density_matrix = UHF._build_total_density_matrix
    _get_occupation = UHF._get_occupation
    _print_orbital_energies = UHF._print_orbital_energies
    _assign_orbital_symmetries = UHF._assign_orbital_symmetries
    _print_orbital_energies = UHF._print_orbital_energies
    _print_ao_composition = UHF._print_ao_composition
    _apply_level_shift = UHF._apply_level_shift

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _build_fock(self, H, fock_builder, S):
        F, _ = UHF._build_fock(self, H, fock_builder, S)

        F_canon = self._build_canonical_fock(F, S)

        return F, F_canon

    def _build_canonical_fock(self, F, S):

        # Projection matrices for core (doubly occupied), open (singly occupied, alpha), and virtual (unoccupied) MO spaces
        U_core = np.dot(self.D[1], S)
        U_open = np.dot(self.D[0] - self.D[1], S)
        U_virt = np.eye(self.nbf) - np.dot(self.D[0], S)

        # Closed-shell Fock
        F_cs = 0.5 * (F[0] + F[1])

        def _project(u, v, f):
            return np.einsum("ur,uv,vt->rt", u, f, v, optimize=True)

        def _build_fock_eff(f):
            # these are scaled by 0.5 to account for fock + fock.T below
            fock = _project(U_core, U_core, f) * 0.5  # cc
            fock += _project(U_open, U_open, f) * 0.5  # oo
            fock += _project(U_virt, U_virt, f) * 0.5  # vv
            # off-diagonal blocks
            fock += _project(U_open, U_core, f)  # oc
            fock += _project(U_open, U_virt, f)  # ov
            fock += _project(U_virt, U_core, F_cs)  # Replace cv sector with fcore
            fock = fock + fock.conj().T
            return fock

        return [_build_fock_eff(f) for f in F]


@dataclass
class GHF(SCFBase):
    r"""
    Generalized Hartree-Fock (GHF) method.
    The GHF spinor basis is a direct product of the atomic basis and :math:`\{|\alpha\rangle, |\beta\rangle\}`:
    
    .. math::
    
        |\psi_i\rangle = \sum_{\mu} \sum_{\sigma\in\{\alpha,\beta\}} c^{\sigma}_{\mu i} |\chi_{\mu}\rangle\otimes|\sigma\rangle

    The MO coefficients are stored in a square array

    .. math::
        \mathbf{C} = \begin{bmatrix}
        \mathbf{c}^{\alpha}_0 & \mathbf{c}^{\alpha}_1 & \dots\\
        \mathbf{c}^{\beta}_0 & \mathbf{c}^{\beta}_1 & \dots
        \end{bmatrix}

    Parameters
    ----------
    guess_mix : bool, optional, default=False
        If True, will mix the HOMO and LUMO orbitals to try to break alpha-beta degeneracy if nel is even.
    alpha_beta_mix : bool, optional, default=False
        If True, will mix the highest two spinorbitals to try to seed alpha-beta orbital gradients.
    break_complex_symmetry : bool, optional, default=False
        If True, will add a small complex perturbation to the initial density matrix. This will both break
        the complex conjugation symmetry and Sz symmetry (allowing alpha-beta density matrix blocks to be nonzero)
    j_adapt: bool, optional, default=False
        If True, the j-adapted spinor AO basis will be used instead of the spherical AO basis.
    """

    ms_guess: float = None
    guess_mix: bool = False  # only used if nel is even
    alpha_beta_mix: bool = False
    break_complex_symmetry: bool = False
    j_adapt: bool = False

    _diis_update = RHF._diis_update

    def __call__(self, system):
        system.two_component = True
        if self.j_adapt:
            ua, ub = real_sph_to_j_adapted(system.basis)
            self.Usph2j = np.vstack((ua, ub))
            S = system.ints_overlap()
            S_spinor = self.Usph2j.conj().T @ S @ self.Usph2j
            self.Xorth_spinor, _ = compute_orthonormal_transformation(
                S_spinor,
                system.linear_dep_trigger,
                system.ortho_thresh,
            )
        self = super().__call__(system)
        self._parse_state()
        return self

    def _parse_state(self):
        if self.ms_guess is not None:
            assert np.isclose(
                int(round(self.ms_guess * 2)), self.ms_guess * 2
            ), "ms_guess must be a multiple of 0.5."
            self.twicems_guess = int(round(self.ms_guess * 2))
            if self.nel % 2 != self.twicems_guess % 2:
                raise ValueError(
                    f"{self.nel} electrons is incompatible with ms_guess={self.ms_guess}!"
                )
            self.na_guess = int(round(self.nel + self.twicems_guess) / 2)
            self.nb_guess = int(round(self.nel - self.twicems_guess) / 2)
            assert (
                self.nel == self.na_guess + self.nb_guess
            ), f"Number of electrons {self.nel} does not match na + nb = {self.na_guess} + {self.nb_guess}."
            assert (
                self.na_guess >= 0 and self.nb_guess >= 0
            ), f"{self._scf_type} requires non-negative number of alpha and beta electrons."

    def _build_fock(self, H, fock_builder, S):
        Jaa, Jbb = fock_builder.build_J([self.D[0], self.D[3]])
        nbf = Jaa.shape[0]
        if self.iter == 0 and self.ms_guess is not None:
            # Apply na/nb_guess
            mo_a, mo_b = self._guess_ms(self.C[0])
            occ = list(mo_a[: self.na_guess]) + list(mo_b[: self.nb_guess])
            occ = sorted(occ)
            Kaa, Kab, Kba, Kbb = fock_builder.build_K([self.C[0][:, occ]])
        else:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K([self.C[0][:, : self.nel]])
        F = H.copy()
        F[:nbf, :nbf] += Jaa + Jbb - Kaa
        F[:nbf, nbf:] += -Kab
        F[nbf:, :nbf] += -Kba
        F[nbf:, nbf:] += Jaa + Jbb - Kbb

        F_canon = F

        return [F], [F_canon]

    def _build_density_matrix(self):
        # D = Cocc Cocc^+
        if self.iter == 0 and self.ms_guess is not None:
            # apply na/nb_guess
            occ_a, occ_b = self._guess_ms(self.C[0])
            Ca = self.C[0][:, occ_a[: self.na_guess]]
            Cb = self.C[0][:, occ_b[: self.nb_guess]]
            D = np.einsum("mi,ni->mn", Ca, Ca.conj())
            D += np.einsum("mi,ni->mn", Cb, Cb.conj())
        else:
            D = np.einsum(
                "mi,ni->mn",
                self.C[0][:, : self.nel],
                self.C[0][:, : self.nel].conj(),
            )
        nbf = self.nbf
        Daa = D[:nbf, :nbf]
        Dab = D[:nbf, nbf:]
        Dba = D[nbf:, :nbf]
        Dbb = D[nbf:, nbf:]

        return Daa, Dab, Dba, Dbb

    def _build_total_density_matrix(self):
        Daa, *_, Dbb = self._build_density_matrix()
        return Daa + Dbb

    def _apply_level_shift(self, F, S):
        if self.level_shift is None or self.level_shift < 1e-4:
            return F
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        D_vir = S - S @ D_spinor @ S

        return [F[0] + self.level_shift * D_vir]

    def _initial_guess(self, H, guess_type="minao"):
        C = RHF._initial_guess(self, H, guess_type)[0]
        if self.guess_mix and self.ms_guess is not None:
            if self.ms_guess == 0:
                C = guess_mix(C, self.nel - 1, twocomp=True)
        if self.alpha_beta_mix:
            C = alpha_beta_mix(C)
        if self.break_complex_symmetry:
            C = break_complex_conjugation_symmetry(C)
        return [C]

    def _build_ao_grad(self, S, F):
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        sdf = S @ D_spinor @ F[0]
        AO_grad = sdf.conj().T - sdf
        AO_grad = self.Xorth.T @ AO_grad @ self.Xorth

        return AO_grad

    def _eigh(self, F):
        Xorth = self.Xorth_spinor if self.j_adapt else self.Xorth
        Ftilde = Xorth.conj().T @ F @ Xorth
        e, c = np.linalg.eigh(Ftilde)
        return e, Xorth @ c

    def _diagonalize_fock(self, F):
        if self.j_adapt:
            F_spinor = self.Usph2j.conj().T @ F[0] @ self.Usph2j
            eps, C = self._eigh(F_spinor)
            C = self.Usph2j @ C
        else:
            eps, C = self._eigh(F[0])
        return [eps], [C]

    def _spin(self, S):
        """
        S^2 = 0.5 * (S+S- + S-S+) + Sz^2, S+ = sum_i si+, S- = sum_i si-
        We make use of the Slater-Condon rules to compute <GHF|S^2|GHF>
        """
        S_1c = S[: self.nbf, : self.nbf]
        mo_a = self.C[0][: self.nbf, : self.nel]
        mo_b = self.C[0][self.nbf :, : self.nel]

        # MO basis overlap matrices
        saa = mo_a.conj().T @ S_1c @ mo_a
        sbb = mo_b.conj().T @ S_1c @ mo_b
        sab = mo_a.conj().T @ S_1c @ mo_b
        sba = sab.conj().T

        na = saa.trace()
        nb = sbb.trace()

        S2_diag = (na + nb) * 0.5
        S2_offdiag = sba.trace() * sab.trace() - np.einsum("ij,ji->", sba, sab)
        Sz2_diag = (na + nb) * 0.25
        Sz2_offdiag = 0.25 * ((na - nb) ** 2 - np.linalg.norm(saa - sbb) ** 2)
        return (S2_diag + S2_offdiag + Sz2_diag + Sz2_offdiag).real

    def _energy(self, H, F):
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        energy = 0.5 * np.einsum("vu,uv->", D_spinor, H) + 0.5 * np.einsum(
            "vu,uv->", D_spinor, F[0]
        )
        return energy.real

    def _get_occupation(self):
        self.nocc = self.nel
        self.nuocc = self.nmo * 2 - self.nocc

    def _print_orbital_energies(self):
        nocc = self.nel
        nuocc = self.nmo * 2 - nocc
        orb_per_row = 5
        logger.log_info1("---------------------")
        logger.log_info1("Spinor Energies [Eh]")
        logger.log_info1("---------------------")
        logger.log_info1("Occupied:")
        string = ""
        for i in range(nocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = nocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
            )
        logger.log_info1(string)

    def _assign_orbital_symmetries(self):
        S = self._get_overlap()
        self.irrep_labels, self.irrep_indices = assign_mo_symmetries(
            self.system, S, self.C[0]
        )

    def _guess_ms(self, C):
        nmo = C.shape[1]
        nbf = C.shape[0] // 2
        mo_a = []
        mo_b = []
        for i in range(nmo):
            norm_a = np.linalg.norm(C[:nbf, i])
            norm_b = np.linalg.norm(C[nbf:, i])
            if norm_a >= norm_b:
                mo_a.append(i)
            else:
                mo_b.append(i)
        return np.array(mo_a), np.array(mo_b)

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        if self.system.x2c_type == "so":
            logger.log_info1("\nSpinor Composition of MOs (HOMO-6 to HOMO):")
            if not hasattr(self, "Usph2j"):
                ua, ub = real_sph_to_j_adapted(self.system.basis)
                self.Usph2j = np.vstack((ua, ub))
            C = self.Usph2j.conj().T @ self.C[0]
            basis_info.print_spinor_composition(
                C, list(range(max(self.nel - 6, 0), self.nel))
            )
            logger.log_info1("\nSpinor Composition of MOs (LUMO to LUMO+6):")
            basis_info.print_spinor_composition(
                C, list(range(self.nel, min(self.nel + 6, self.nmo * 2)))
            )
        else:
            logger.log_info1("\nAO Composition of MOs (HOMO-5 to HOMO):")
            basis_info.print_ao_composition(
                self.C[0], list(range(max(self.nel - 6, 0), self.nel)), spinorbital=True
            )
            logger.log_info1("\nAO Composition of MOs (LUMO to LUMO+5):")
            basis_info.print_ao_composition(
                self.C[0],
                list(range(self.nel, min(self.nel + 6, self.nmo * 2))),
                spinorbital=True,
            )
