from dataclasses import dataclass, field
import time
import numpy as np
import scipy as sp

import forte2
from forte2.jkbuilder import FockBuilder
from forte2.helpers.mixins import MOsMixin
from forte2.helpers.matrix_functions import givens_rotation, eigh_gen, canonical_orth
from forte2.helpers import logger
from .initial_guess import minao_initial_guess, core_initial_guess


def guess_mix(C, homo_idx, mixing_parameter=np.pi / 4):
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    Ca = givens_rotation(C, cosq, sinq, homo_idx, homo_idx + 1)
    Cb = givens_rotation(C, cosq, -sinq, homo_idx, homo_idx + 1)
    return [Ca, Cb]


@dataclass
class SCFMixin:
    charge: int
    diis_start: int = 4
    diis_nvec: int = 8
    econv: float = 1e-9
    dconv: float = 1e-6
    maxiter: int = 100
    guess_type: str = "minao"

    executed: bool = field(default=False, init=False)
    converged: bool = field(default=False, init=False)

    def __call__(self, system):
        self.system = system
        self.method = self._scf_type().upper()
        self.nel = self.system.Zsum - self.charge
        assert self.nel >= 0, "Number of electrons must be non-negative."

        if self.method != "GHF" and self.system.x2c_type == "so":
            raise ValueError(
                "SO-X2C is only available for GHF. Use SF-X2C for RHF/UHF."
            )

        self.C = None
        self.Xorth = self.system.Xorth

        return self

    def _eigh(self, F):
        Ftilde = self.Xorth.T @ F @ self.Xorth
        e, c = np.linalg.eigh(Ftilde)
        return e, self.Xorth @ c

    def _eigh_spinor(self, F):
        Xorth = sp.linalg.block_diag(self.Xorth, self.Xorth)
        Ftilde = Xorth.T @ F @ Xorth
        e, c = np.linalg.eigh(Ftilde)
        return e, Xorth @ c

    def _scf_type(self):
        return type(self).__name__.upper()

    def run(self):
        """
        Run the SCF calculation.
        Returns:
            self: The SCF object.
        """
        start = time.monotonic()

        diis = forte2.helpers.DIIS(diis_start=self.diis_start, diis_nvec=self.diis_nvec)
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

        if self.C is None:
            self.C = self._initial_guess(H, S, guess_type=self.guess_type)
        self.D = self._build_initial_density_matrix()

        Eold = 0.0
        Dold = self.D

        width = 109
        logger.log_info1("=" * width)
        logger.log_info1(
            f"{'Iter':>4s} {'Energy':>20s} {'deltaE':>20s} {'|deltaD|':>20s} {'|AO grad|':>20s} {'<S^2>':>20s}"
        )
        logger.log_info1("-" * width)
        self.iter = 0
        for iter in range(self.maxiter):

            # 1. Build the Fock matrix
            # (there is a slot for canonicalized F to accommodate ROHF and CUHF methods - admittedly weird for RHF/UHF)
            F, F_canon = self._build_fock(H, fock_builder, S)
            # 2. Build the orbital gradient (DIIS residual)
            AO_grad = self._build_ao_grad(S, F_canon)
            # 3. Perform DIIS update of Fock
            F_canon = self._diis_update(diis, F_canon, AO_grad)
            # 4. Diagonalize updated Fock
            self.eps, self.C = self._diagonalize_fock(F_canon)
            # 5. Build new density matrix
            self.D = self._build_density_matrix()
            # 6. Compute new HF energy
            self.E = Vnn + self._energy(H, F)

            # check convergence parameters
            deltaE = self.E - Eold
            deltaD = sum([np.linalg.norm(d - dold) for d, dold in zip(self.D, Dold)])
            self.S2 = self._spin(S)

            # print iteration
            logger.log_info1(
                f"{iter + 1:4d} {self.E:20.12f} {deltaE:20.12f} {deltaD:20.12f} {np.linalg.norm(AO_grad):20.12f} {self.S2:20.12f}"
            )

            if np.abs(deltaE) < self.econv and deltaD < self.dconv:
                logger.log_info1("=" * width)
                logger.log_info1(f"{self.method} iterations converged\n")
                # perform final iteration
                F, F_canon = self._build_fock(H, fock_builder, S)
                self.eps, self.C = self._diagonalize_fock(F_canon)
                self.D = self._build_density_matrix()
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
            raise RuntimeError(
                f"{self.method} did not converge in {self.maxiter} iterations."
            )

        end = time.monotonic()
        logger.log_info1(f"{self.method} time: {end - start:.2f} seconds")

        if self.converged:
            self._post_process()

        self.executed = True
        return self

    def _get_hcore(self):
        return self.system.ints_hcore()

    def _get_overlap(self):
        return self.system.ints_overlap()

    def _get_nuclear_repulsion(self):
        return self.system.nuclear_repulsion_energy()

    def _post_process(self):
        """
        Post-process the SCF results.
        This method can be overridden by subclasses to perform additional calculations.
        """
        self._print_orbital_energies()


@dataclass
class RHF(SCFMixin, MOsMixin):

    def __call__(self, system):
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
        return F, F

    def _build_density_matrix(self):
        D = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        return [D]

    def _build_total_density_matrix(self):
        # returns the total density matrix (Daa + Dbb)
        return 2 * self._build_density_matrix()[0]

    def _initial_guess(self, H, S, guess_type="minao"):
        match guess_type:
            case "minao":
                C = minao_initial_guess(self.system, H, S)
            case "hcore":
                C = core_initial_guess(self.system, H, S)
            case _:
                raise RuntimeError(f"Unknown initial guess type: {guess_type}")

        return [C]

    def _build_initial_density_matrix(self):
        return self._build_density_matrix()

    def _build_ao_grad(self, S, F):
        ao_grad = F @ self.D[0] @ S - S @ self.D[0] @ F
        ao_grad = self.Xorth.T @ ao_grad @ self.Xorth
        return ao_grad

    def _energy(self, H, F):
        return np.sum(self.D[0] * (H + F))

    def _diagonalize_fock(self, F):
        eps, C = self._eigh(F)
        return [eps], [C]

    def _spin(self, S):
        return self.ms * (self.ms + 1)

    def _diis_update(self, diis, F, AO_grad):
        return diis.update(F, AO_grad)

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
            string += f"{i+1:<4d} {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = ndocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx+1:<4d} {self.eps[0][idx]:<12.6f} "
        logger.log_info1(string)


@dataclass
class UHF(SCFMixin, MOsMixin):
    ms: float = field(default=None, init=True)
    guess_mix: bool = False  # only used if ms == 0

    def __call__(self, system):
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

    def _initial_guess(self, H, S, guess_type="minao"):
        C = RHF._initial_guess(self, H, S, guess_type=guess_type)[0]

        if self.twicems == 0 and self.guess_mix:
            return guess_mix(C, self.nel // 2 - 1)

        return [C, C]

    def _build_initial_density_matrix(self):
        D_a, D_b = self._build_density_matrix()
        return D_a, D_b

    def _build_ao_grad(self, S, F):
        AO_grad = np.hstack(
            [(f @ d @ S - S @ d @ f).flatten() for d, f in zip(self.D, F)]
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
            string += f"{i+1:<4d} {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nAlpha Virtual:")
        string = ""
        for i in range(naucc):
            idx = naocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx+1:<4d} {self.eps[0][idx]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nBeta Occupied:")
        string = ""
        for i in range(nbocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} {self.eps[1][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nBeta Virtual:")
        string = ""
        for i in range(nbucc):
            idx = nbocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx+1:<4d} {self.eps[1][i]:<12.6f} "
        logger.log_info1(string)


@dataclass
class ROHF(SCFMixin, MOsMixin):
    ms: float = field(default=None, init=True)

    _parse_state = UHF._parse_state
    _initial_guess = RHF._initial_guess
    _build_initial_density_matrix = UHF._build_initial_density_matrix
    _diagonalize_fock = RHF._diagonalize_fock
    _spin = RHF._spin
    _energy = UHF._energy
    _diis_update = RHF._diis_update
    _build_total_density_matrix = UHF._build_total_density_matrix

    def __call__(self, system):
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
        return fock

    def _build_density_matrix(self):
        D_a = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        D_b = np.einsum("mi,ni->mn", self.C[0][:, : self.nb], self.C[0][:, : self.nb])
        return D_a, D_b

    def _build_ao_grad(self, S, F):
        Deff = 0.5 * (self.D[0] + self.D[1])
        return F @ Deff @ S - S @ Deff @ F

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
            string += f"{i+1:<4d} {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        if nsocc > 0:
            logger.log_info1("\nSingly Occupied:")
            string = ""
            for i in range(nsocc):
                idx = ndocc + i
                if i % orb_per_row == 0:
                    string += "\n"
                string += f"{idx+1:<4d} {self.eps[0][idx]:<12.6f} "
            logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = ndocc + nsocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx+1:<4d} {self.eps[0][idx]:<12.6f} "
        logger.log_info1(string)


@dataclass
class CUHF(SCFMixin, MOsMixin):
    ms: float = field(default=None, init=True)
    guess_mix: bool = False  # only used if ms == 0

    _parse_state = UHF._parse_state
    _build_density_matrix = UHF._build_density_matrix
    _initial_guess = UHF._initial_guess
    _build_initial_density_matrix = UHF._build_initial_density_matrix
    _build_ao_grad = UHF._build_ao_grad
    _diagonalize_fock = UHF._diagonalize_fock
    _spin = UHF._spin
    _energy = UHF._energy
    _diis_update = UHF._diis_update
    _build_total_density_matrix = UHF._build_total_density_matrix
    _print_orbital_energies = UHF._print_orbital_energies

    def __call__(self, system):
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
class GHF(SCFMixin, MOsMixin):
    r"""
    Generalized Hartree-Fock (GHF) method.
    The GHF spinor basis is a direct product of the atomic basis and :math:`\{|\alpha\rangle, |\beta\rangle\}`:
    
    .. math::
    
        |\psi_i\rangle = \sum_{\mu} \sum_{\sigma\in\{\alpha,\beta\}} c_{\mu,\sigma} |\chi_{\mu}\rangle|\sigma\rangle

    The MO coefficients are stored in a square array

    .. math::
        C = \begin{bmatrix}
        C_{\alpha,i} \\
        C_{\beta,i}
        \end{bmatrix}
    """

    guess_mix: bool = False  # only used if nel is even
    break_complex_symmetry: bool = False

    _diis_update = RHF._diis_update

    def __call__(self, system):
        self = super().__call__(system)
        if self.system.x2c_type == "sf" or self.system.x2c_type == None:
            H = self._get_hcore().astype(complex)
            self._get_hcore = lambda: sp.linalg.block_diag(H, H)
        return self

    def _build_fock(self, H, fock_builder, S):
        Jaa, Jbb = fock_builder.build_J([self.D[0], self.D[3]])
        nbf = Jaa.shape[0]
        if self.iter == 0 and self.break_complex_symmetry:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K_density(self.D)
        else:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K(
                [self.C[0][:nbf, : self.nel], self.C[0][nbf:, : self.nel]], cross=True
            )
        F = H.copy()
        F[:nbf, :nbf] += Jaa + Jbb - Kaa
        F[:nbf, nbf:] += -Kab
        F[nbf:, :nbf] += -Kba
        F[nbf:, nbf:] += Jaa + Jbb - Kbb

        F_canon = F

        return F, F_canon

    def _build_density_matrix(self):
        # D = Cocc Cocc^+
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

    def _initial_guess(self, H, S, guess_type="minao"):
        H_ao = forte2.ints.kinetic(self.system.basis) + forte2.ints.nuclear(
            self.system.basis, self.system.atoms
        )
        Ca = Cb = RHF._initial_guess(self, H_ao, S, guess_type)[0].astype(complex)
        if self.nel % 2 == 0 and self.guess_mix:
            Ca, Cb = guess_mix(Ca, self.nel // 2 - 1)
        C_spinor = np.zeros((self.nbf * 2,) * 2, dtype=complex)
        C_spinor[: self.nbf, ::2] = Ca
        C_spinor[self.nbf :, 1::2] = Cb
        return [C_spinor]

    def _build_initial_density_matrix(self):
        Daa, Dab, Dba, Dbb = self._build_density_matrix()

        if self.break_complex_symmetry:
            Daa[0, :] += 0.05j
            Dab[0, :] += 0.05j
            Daa[:, 0] -= 0.05j
            Dba[:, 0] -= 0.05j
            Dbb[0, :] += 0.05j
            Dbb[:, 0] -= 0.05j

        return Daa, Dab, Dba, Dbb

    def _build_ao_grad(self, S, F):
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        S_spinor = np.block([[S, np.zeros_like(S)], [np.zeros_like(S), S]])
        sdf = S_spinor @ D_spinor @ F
        AO_grad = sdf.conj().T - sdf

        return AO_grad

    def _diagonalize_fock(self, F):
        eps, C = self._eigh_spinor(F)
        return [eps], [C]

    def _spin(self, S):
        """
        S^2 = 0.5 * (S+S- + S-S+) + Sz^2, S+ = sum_i si+, S- = sum_i si-
        We make use of the Slater-Condon rules to compute <GHF|S^2|GHF>
        """
        mo_a = self.C[0][: self.nbf, : self.nel]
        mo_b = self.C[0][self.nbf :, : self.nel]

        # MO basis overlap matrices
        saa = mo_a.conj().T @ S @ mo_a
        sbb = mo_b.conj().T @ S @ mo_b
        sab = mo_a.conj().T @ S @ mo_b
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
            "vu,uv->", D_spinor, F
        )
        return energy.real

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
            string += f"{i+1:<4d} {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = nocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx+1:<4d} {self.eps[0][idx]:<12.6f} "
        logger.log_info1(string)
