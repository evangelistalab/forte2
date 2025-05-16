from dataclasses import dataclass
import time
import numpy as np
import scipy as sp

import forte2
from forte2.jkbuilder.jkbuilder import DFFockBuilder
from forte2.helpers.mixins import MOs
from .initial_guess import minao_initial_guess


class SCFMixin:
    def _check_scf_params(self):
        pass

    def _scf_type(self):
        return type(self).__name__.upper()

    def _check_parameters(self):
        assert self.mult > 0
        assert self.charge <= self.nel
        scftyp = self._scf_type()
        if scftyp == "RHF":
            assert self.na == self.nb

    def run(
        self,
        system,
        diis_start=4,
        diis_nvec=8,
        econv=1e-6,
        dconv=1e-3,
        maxiter=100,
        **kwargs,
    ):
        start = time.monotonic()

        method = self._scf_type().upper()
        Zsum = np.sum([x[0] for x in system.atoms])
        self.nel = Zsum - self.charge
        self.nbasis = system.basis.size
        self.naux = system.auxiliary_basis.size
        self.na = (self.nel + self.mult - 1) // 2
        self.nb = (self.nel - self.mult + 1) // 2
        self.sz = (self.na - self.nb) / 2.0

        self._check_parameters()

        diis = forte2.helpers.DIIS(diis_start=diis_start, diis_nvec=diis_nvec)

        print(f"Number of alpha electrons: {self.na}")
        print(f"Number of beta electrons: {self.nb}")
        print(f"Total charge: {self.charge}")
        print(f"Spin multiplicity: {self.mult}")
        print(f"Number of basis functions: {self.nbasis}")
        print(f"Number of auxiliary basis functions: {self.naux}")
        print(f"Energy convergence criterion: {econv:e}")
        print(f"Density convergence criterion: {dconv:e}")
        print(f"DIIS acceleration: {diis.do_diis}")
        print(f"\n==> {method} SCF ROUTINE <==")

        Vnn = self._get_nuclear_repulsion(system.atoms)
        S = self._get_overlap(system.basis)
        H = self._get_hcore(system)
        fock_builder = DFFockBuilder(system)

        self.C = self._initial_guess(system, H, S)
        self.D = self._build_initial_density_matrix(**kwargs)

        Eold = 0.0
        Dold = self.D

        width = 109
        print("=" * width)
        print(
            f"{'Iter':>4s} {'Energy':>20s} {'deltaE':>20s} {'|deltaD|':>20s} {'|AO grad|':>20s} {'<S^2>':>20s}"
        )
        print("-" * width)
        self.iter = 0
        for iter in range(maxiter):

            # 1. Build the Fock matrix
            # (there is a slot for canonicalized F to accommodate ROHF and CUHF methods - admittedly weird for RHF/UHF)
            F, F_canon = self._build_fock(H, fock_builder, S)
            # 2. Build the orbital gradient (DIIS residual)
            AO_grad = self._build_ao_grad(S, F_canon)
            # 3. Perform DIIS update of Fock
            F_canon = self._diis_update(diis, F_canon, AO_grad)
            # 4. Diagonalize updated Fock
            self.eps, self.C = self._diagonalize_fock(F_canon, S)
            # 5. Build new density matrix
            self.D = self._build_density_matrix()
            # 6. Compute new HF energy
            self.E = Vnn + self._energy(H, F)

            # check convergence parameters
            deltaE = self.E - Eold
            deltaD = sum([np.linalg.norm(d - dold) for d, dold in zip(self.D, Dold)])
            self.S2 = self._spin(S)

            # print iteration
            print(
                f"{iter + 1:4d} {self.E:20.12f} {deltaE:20.12f} {deltaD:20.12f} {np.linalg.norm(AO_grad):20.12f} {self.S2:20.12f}"
            )

            if np.abs(deltaE) < econv and deltaD < dconv:
                print("-" * width)
                print(f"{method} iterations converged\n")
                # perform final iteration
                F, F_canon = self._build_fock(H, fock_builder, S)
                self.eps, self.C = self._diagonalize_fock(F_canon, S)
                self.D = self._build_density_matrix()
                self.E = Vnn + self._energy(H, F)
                print(f"Final {method} Energy: {self.E:20.12f}")
                break

            # reset old parameters
            Eold = self.E
            Dold = self.D
            self.iter += 1
        else:
            print("-" * width)
            print(f"{method} iterations not converged!")

        end = time.monotonic()
        print(f"{method} time: {end - start:.2f} seconds")

        return self

    def x2c(self, x2c_type="sf", **kwargs):
        """
        X2C transformation for the RHF/UHF methods.
        """
        from forte2.scf.x2c import get_hcore_x2c

        assert x2c_type in [
            "sf",
            "so",
        ], f"Invalid x2c_type: {x2c_type}. Must be 'sf' or 'so'."

        if self._scf_type() != "GHF" and x2c_type == "so":
            raise ValueError(
                "SO-X2C is only available for GHF. Use SF-X2C for RHF/UHF."
            )

        if x2c_type == "sf":
            if self._scf_type() in ["RHF", "UHF", "CUHF"]:
                self._get_hcore = lambda system: get_hcore_x2c(system, x2c_type="sf")
            elif self._scf_type() == "GHF":

                def _get_hcore(system):
                    H = get_hcore_x2c(system, x2c_type="sf")
                    return sp.linalg.block_diag(H, H)

                self._get_hcore = _get_hcore
        elif x2c_type == "so":
            self._get_hcore = lambda system: get_hcore_x2c(system, x2c_type="so")

        return self


@dataclass
class RHF(SCFMixin, MOs):

    charge: int
    mult: int = 1

    def _get_hcore(self, system):
        T = forte2.ints.kinetic(system.basis)
        V = forte2.ints.nuclear(system.basis, system.atoms)
        H = T + V
        return H

    def _get_overlap(self, basis):
        S = forte2.ints.overlap(basis)
        return S

    def _get_nuclear_repulsion(self, atoms):
        Vnn = forte2.ints.nuclear_repulsion(atoms)
        return Vnn

    def _build_fock(self, H, fock_builder, S):
        J = fock_builder.build_J(self.D)[0]
        K = fock_builder.build_K([self.C[0][:, : self.na]])[0]
        F = H + 2.0 * J - K
        return F, F

    def _build_density_matrix(self):
        D = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        return [D]

    def _initial_guess(self, system, H, S):
        C = minao_initial_guess(system, H, S)
        return [C]

    def _build_initial_density_matrix(self, **kwargs):
        return self._build_density_matrix()

    def _build_ao_grad(self, S, F):
        ao_grad = F @ self.D[0] @ S - S @ self.D[0] @ F
        return ao_grad

    def _energy(self, H, F):
        return np.sum(self.D[0] * (H + F))

    def _diagonalize_fock(self, F, S):
        eps, C = sp.linalg.eigh(F, S)
        return [eps], [C]

    def _spin(self, S):
        return self.sz * (self.sz + 1)

    def _diis_update(self, diis, F, AO_grad):
        return diis.update(F, AO_grad)


@dataclass
class UHF(SCFMixin, MOs):
    charge: int
    mult: int

    _get_hcore = RHF._get_hcore
    _get_overlap = RHF._get_overlap
    _get_nuclear_repulsion = RHF._get_nuclear_repulsion

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

    def _initial_guess(self, system, H, S):
        C = minao_initial_guess(system, H, S)
        return [C, C]

    def _build_initial_density_matrix(self, **kwargs):
        D_a, D_b = self._build_density_matrix()
        if self.mult != 1:
            D_b *= 0.0
        return D_a, D_b

    def _build_ao_grad(self, S, F):
        AO_grad = np.hstack(
            [(f @ d @ S - S @ d @ f).flatten() for d, f in zip(self.D, F)]
        )
        return AO_grad

    def _diagonalize_fock(self, F, S):
        eps_a, C_a = sp.linalg.eigh(F[0], S)
        eps_b, C_b = sp.linalg.eigh(F[1], S)
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
        s2 = self.sz * (self.sz + 1) + ds2
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
            F_flat[: self.nbasis**2].reshape(self.nbasis, self.nbasis),
            F_flat[self.nbasis**2 :].reshape(self.nbasis, self.nbasis),
        ]
        return F


@dataclass
class ROHF(SCFMixin, MOs):
    charge: int
    mult: int

    _get_hcore = RHF._get_hcore
    _get_overlap = RHF._get_overlap
    _get_nuclear_repulsion = RHF._get_nuclear_repulsion
    _initial_guess = RHF._initial_guess
    _build_initial_density_matrix = UHF._build_initial_density_matrix
    _diagonalize_fock = RHF._diagonalize_fock
    _spin = RHF._spin
    _energy = UHF._energy
    _diis_update = RHF._diis_update

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
        U_virt = np.eye(self.nbasis) - np.dot(self.D[0], S)

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


@dataclass
class CUHF(SCFMixin, MOs):
    charge: int
    mult: int

    _get_hcore = RHF._get_hcore
    _get_overlap = RHF._get_overlap
    _get_nuclear_repulsion = RHF._get_nuclear_repulsion
    _build_density_matrix = UHF._build_density_matrix
    _initial_guess = UHF._initial_guess
    _build_initial_density_matrix = UHF._build_initial_density_matrix
    _build_ao_grad = UHF._build_ao_grad
    _diagonalize_fock = UHF._diagonalize_fock
    _spin = UHF._spin
    _energy = UHF._energy
    _diis_update = UHF._diis_update

    def _build_fock(self, H, fock_builder, S):
        F, _ = UHF._build_fock(self, H, fock_builder, S)

        F_canon = self._build_canonical_fock(F, S)

        return F, F_canon

    def _build_canonical_fock(self, F, S):

        # Projection matrices for core (doubly occupied), open (singly occupied, alpha), and virtual (unoccupied) MO spaces
        U_core = np.dot(self.D[1], S)
        U_open = np.dot(self.D[0] - self.D[1], S)
        U_virt = np.eye(self.nbasis) - np.dot(self.D[0], S)

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
class GHF(SCFMixin, MOs):
    """
    Generalized Hartree-Fock (GHF) method.
    The GHF spinor basis is a direct product of the atomic basis and {|\alpha>, |\beta>}
    |\psi_i> = \sum_{\mu} \sum_{\sigma\in\{\alpha,\beta\}} c_{\mu\sigma} |\chi_{\mu}>|\sigma>
    The MO coefficients are stored in a square array
    [------MOs------]
    [alpha basis    ]
    [               ]
    [               ]
    [beta basis     ]
    [               ]
    """

    charge: int
    mult: int = 1

    _get_hcore_ao = RHF._get_hcore
    _get_overlap = RHF._get_overlap
    _get_nuclear_repulsion = RHF._get_nuclear_repulsion
    _diis_update = RHF._diis_update

    def _get_hcore(self, system):
        H = RHF._get_hcore(self, system).astype(complex)
        return sp.linalg.block_diag(H, H)

    def _build_fock(self, H, fock_builder, S):
        Jaa, Jbb = fock_builder.build_J([self.D[0], self.D[3]])
        nbf = Jaa.shape[0]
        if self.iter == 0:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K_density(self.D)
        else:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K(
                [self.C[0][:nbf, : self.nel], self.C[0][nbf:, : self.nel]], ghf=True
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
        nbf = self.nbasis
        Daa = D[:nbf, :nbf]
        Dab = D[:nbf, nbf:]
        Dba = D[nbf:, :nbf]
        Dbb = D[nbf:, nbf:]
        return Daa, Dab, Dba, Dbb

    def _initial_guess(self, system, H, S):
        H_ao = self._get_hcore_ao(system)
        return [minao_initial_guess(system, H_ao, S).astype(complex)]

    def _build_initial_density_matrix(self, **kwargs):
        break_spin_symmetry = kwargs.get("break_spin_symmetry", True)
        break_complex_symmetry = kwargs.get("break_complex_symmetry", True)

        Daa = np.einsum(
            "mi,ni->mn",
            self.C[0][:, : self.nel // 2],
            self.C[0][:, : self.nel // 2].conj(),
        )
        Dab = np.zeros_like(Daa)
        Dba = np.zeros_like(Daa)
        Dbb = Daa.copy()

        if break_spin_symmetry:
            diag = np.diag(Daa)
            Dab = np.diag(diag) * 0.05
            Dba = np.diag(diag) * 0.05
        if break_complex_symmetry:
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

    def _diagonalize_fock(self, F, S):
        S_spinor = np.block([[S, np.zeros_like(S)], [np.zeros_like(S), S]])
        eps, C = sp.linalg.eigh(F, S_spinor)
        return [eps], [C]

    def _spin(self, S):
        """
        S^2 = 0.5 * (S+S- + S-S+) + Sz^2, S+ = \sum_i si+, S- = \sum_i si-
        We make use of the Slater-Condon rules to compute <GHF|S^2|GHF>
        """
        mo_a = self.C[0][: self.nbasis, : self.nel]
        mo_b = self.C[0][self.nbasis :, : self.nel]

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
