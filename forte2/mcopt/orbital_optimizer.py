import numpy as np
import scipy as sp

from forte2.jkbuilder import FockBuilder
from forte2.orbitals.orbital_utils import normalize_unitary_matrix


class OrbOptimizer:
    def __init__(
        self,
        C: np.ndarray,
        extents: list[slice],
        fock_builder: FockBuilder,
        hcore: np.ndarray,
        e_nuc: float,
        nrr: np.ndarray,
        compute_active_hessian: bool = False,
    ):
        self.core, self.actv, self.virt = extents
        self.C = C
        self.C0 = C.copy()
        self.Cgen = C
        self.Cact = C[:, self.actv]
        self.Ccore = C[:, self.core]
        self.ncore = self.Ccore.shape[1]
        self.nact = self.Cact.shape[1]
        self.nvirt = self.C.shape[1] - self.ncore - self.nact
        self.fock_builder = fock_builder
        self.hcore = hcore
        self.nrr = nrr
        self.nrot = self.nrr.sum()
        self.e_nuc = e_nuc
        self.compute_active_hessian = compute_active_hessian

        # the skew-hermitian rotation matrix, C_current = C_0 @ exp(R)
        self.R = np.zeros(self.nrot, dtype=float)
        # the unitary transformation matrix, U = exp(R)
        self.U = np.eye(self.C.shape[1], dtype=float)

    def get_eri_gaaa(self):
        self.eri_gaaa = self.fock_builder.two_electron_integrals_gen_block(
            self.Cgen, *(self.Cact,) * 3
        )
        return self.eri_gaaa

    def set_rdms(self, g1, g2):
        self.g1 = g1
        # '2RDM' defined as in [eq (6)]
        self.g2 = 0.5 * (np.einsum("prqs->pqrs", g2) + np.einsum("qrps->pqrs", g2))

    def get_active_space_ints(self):
        """
        Returns the active space integrals.
        """
        return self.eri_gaaa[self.actv, ...]

    def evaluate(self, x):
        do_update_integrals = self._update_orbitals(x)
        if do_update_integrals:
            self._compute_Fcore()
            self.get_eri_gaaa()

        E_orb = self._compute_reference_energy()

        return E_orb

    def gradient(self, x):
        grad = self._compute_orbgrad()
        g = self._mat_to_vec(grad)
        return g

    def hess_diag(self, x):
        hess = self._compute_orbhess()
        h0 = self._mat_to_vec(hess)
        return h0

    def _update_orbitals(self, R):
        dR = R - self.R
        if np.max(np.abs(dR)) < 1e-12:
            # no change in orbitals, skip the update
            return False
        self.R += dR
        self.U = self.U @ self._expm(dR)
        self.U = normalize_unitary_matrix(self.U)

        self.C = self.C0 @ self.U
        self.Cgen = self.C
        self.Ccore = self.C[:, self.core]
        self.Cact = self.C[:, self.actv]
        return True

    def _expm(self, vec):
        M = self._vec_to_mat(vec)
        eM = sp.linalg.expm(M)
        return eM

    def _vec_to_mat(self, x):
        R = np.zeros_like(self.C)
        R[self.nrr] = x
        R += -R.T.conj()
        return R

    def _mat_to_vec(self, R):
        return R[self.nrr]

    def _compute_reference_energy(self):
        energy = self.Ecore + self.e_nuc
        energy += np.einsum("uv,uv->", self.Fcore[self.actv, self.actv], self.g1)
        energy += 0.5 * np.einsum("tvuw,tuvw->", self.get_active_space_ints(), self.g2)
        return energy

    def _compute_Fcore(self):
        # Compute the core Fock matrix [eq (3)], also return the core energy
        Jcore, Kcore = self.fock_builder.build_JK([self.Ccore])
        self.Fcore = np.einsum(
            "mp,mn,nq->pq",
            self.Cgen.conj(),
            self.hcore + 2 * Jcore[0] - Kcore[0],
            self.Cgen,
            optimize=True,
        )
        self.Ecore = np.einsum(
            "pi,qi,pq->",
            self.Ccore.conj(),
            self.Ccore,
            2 * self.hcore + 2 * Jcore[0] - Kcore[0],
        )

    def _compute_Fact(self):
        Jact, Kact = self.fock_builder.build_JK_generalized(self.Cact, self.g1)

        # [eq (13)]
        self.Fact = np.einsum(
            "mp,mn,nq->pq",
            self.Cgen.conj(),
            Jact - 0.5 * Kact,
            self.Cgen,
            optimize=True,
        )

    def _compute_orbgrad(self):
        self._compute_Fact()
        orbgrad = np.zeros_like(self.Fcore)

        self.A_pq = np.zeros_like(self.Fcore)
        self.Fock = self.Fcore + self.Fact

        # compute A_ri (mo, core) block, [eq (10)]
        self.A_pq[:, self.core] = 2.0 * self.Fock[:, self.core]

        # compute A_ru (mo, active) block, [eq (11)]
        self.A_pq[:, self.actv] = np.einsum(
            "rv,vu->ru", self.Fcore[:, self.actv], self.g1
        )
        # (rt|vw) D_tu,vw, where (rt|vw) = <rv|tw>
        self.A_pq[:, self.actv] += np.einsum("rvtw,tuvw->ru", self.eri_gaaa, self.g2)

        # screen small gradients to prevent symmetry breaking
        self.A_pq[np.abs(self.A_pq) < 1e-12] = 0.0

        # compute g_rk (mo, core + active) block of gradient, [eq (9)]
        orbgrad = 2 * (self.A_pq - self.A_pq.T.conj())
        orbgrad *= self.nrr

        return orbgrad

    def _compute_orbhess(self):
        """Diagonal orbital Hessian"""
        orbhess = np.zeros_like(self.Fcore)
        diag_F = np.diag(self.Fock)
        diag_g1 = np.diag(self.g1)
        diag_grad = np.diag(self.A_pq)

        # The VC, VA, AC blocks are based on Theor. Chem. Acc. 97, 88-95 (1997)
        # compute virtual-core block
        orbhess[self.virt, self.core] = 4.0 * (
            diag_F[self.virt, None] - diag_F[None, self.core]
        )

        # compute virtual-active block
        orbhess[self.virt, self.actv] = 2.0 * (
            diag_F[self.virt, None] * diag_g1[None, :] - diag_grad[None, self.actv]
        )

        # compute active-core block
        orbhess[self.actv, self.core] = 4.0 * (
            diag_F[self.actv, None] - diag_F[None, self.core]
        )
        orbhess[self.actv, self.core] += 2.0 * (
            diag_F[None, self.core] * diag_g1[:, None] - diag_grad[self.actv, None]
        )

        # if GAS: compute active-active block [see SI of J. Chem. Phys. 152, 074102 (2020)]
        if self.compute_active_hessian:
            eri_actv = self.get_active_space_ints()
            # A. G^{uu}_{vv}
            Guu_ = np.einsum("uxuy,vvxy->uv", eri_actv, self.g2)
            Guu_ += 2.0 * np.einsum("uuxy,vxvy->uv", eri_actv, self.g2)
            Guu_ += np.diag(self.Fcore)[self.actv, None] * diag_g1[None, :]

            # B. G^{uv}_{vu}
            Guv_ = self.Fcore[self.actv, self.actv] * self.g1.T
            Guv_ += np.einsum("uxvy,vuxy->uv", eri_actv, self.g2)
            Guv_ += 2.0 * np.einsum("uvxy,vxuy->uv", eri_actv, self.g2)

            # compute diagonal hessian
            orbhess[self.actv, self.actv] = 2.0 * (Guu_ + Guu_.T)
            orbhess[self.actv, self.actv] -= 2.0 * (Guv_ + Guv_.T)
            orbhess[self.actv, self.actv] -= 2.0 * (
                diag_grad[self.actv, None] + diag_grad[None, self.actv]
            )
        orbhess *= self.nrr

        return orbhess


class RelOrbOptimizer(OrbOptimizer):
    def __init__(
        self,
        C: np.ndarray,
        extents: list[slice],
        fock_builder: FockBuilder,
        hcore: np.ndarray,
        e_nuc: float,
        nrr: np.ndarray,
        compute_active_hessian: bool = False,
    ):
        super().__init__(
            C,
            extents,
            fock_builder,
            hcore,
            e_nuc,
            nrr,
            compute_active_hessian,
        )
        self.R = self.R.astype(np.complex128)
        self.U = self.U.astype(np.complex128)

    def get_eri_gaaa(self):
        self.eri_gaaa = self.fock_builder.two_electron_integrals_gen_block_spinor(
            self.Cgen, *(self.Cact,) * 3
        )
        return self.eri_gaaa

    def set_rdms(self, g1, g2):
        self.g1 = g1
        # '2RDM' defined as in [eq (6)]
        self.g2 = g2.swapaxes(1, 2)

    def _compute_reference_energy(self):
        energy = self.Ecore + self.e_nuc
        energy += np.einsum("uv,uv->", self.Fcore[self.actv, self.actv], self.g1)
        energy += 0.5 * np.einsum("tvuw,tuvw->", self.get_active_space_ints(), self.g2)
        return energy

    def _compute_Fcore(self):
        # Compute the core Fock matrix [eq (3)], also return the core energy
        Jcore, Kcore = self.fock_builder.build_JK([self.Ccore])
        self.Fcore = np.einsum(
            "mp,nq,mn->pq",
            self.Cgen.conj(),
            self.Cgen,
            self.hcore + Jcore[0] - Kcore[0],
            optimize=True,
        )
        self.Ecore = np.einsum(
            "pi,qi,pq->",
            self.Ccore.conj(),
            self.Ccore,
            self.hcore + 0.5 * (Jcore[0] - Kcore[0]),
        )

    def _compute_Fact(self):
        Jact, Kact = self.fock_builder.build_JK_generalized(self.Cact, self.g1)

        # [eq (13)]
        self.Fact = np.einsum(
            "mp,nq,mn->pq",
            self.Cgen.conj(),
            self.Cgen,
            Jact - Kact,
            optimize=True,
        )

    def _compute_orbgrad(self):
        self._compute_Fact()
        orbgrad = np.zeros_like(self.Fcore)

        self.Fock = np.zeros_like(self.Fcore)
        self.Fock1 = self.Fcore + self.Fact

        # compute A_ri (mo, core) block, [eq (10)]
        self.Fock[self.core, :] += self.Fock1[:, self.core].T

        # compute A_ru (mo, active) block, [eq (11)]
        self.Fock2 = np.zeros_like(self.Fcore)
        self.Fock2[self.actv, :] = np.einsum(
            "tu,qu->tq", self.g1, self.Fcore[:, self.actv], optimize=True
        )
        # (rt|vw) D_tu,vw, where (rt|vw) = <rv|tw>
        self.Fock2[self.actv, :] += np.einsum(
            "tuvw,qvuw->tq", self.g2, self.eri_gaaa, optimize=True
        )
        self.Fock[self.actv, :] += self.Fock2[self.actv, :]

        # screen small gradients to prevent symmetry breaking
        self.Fock[np.abs(self.Fock) < 1e-12] = 0.0

        orbgrad = -2 * (self.Fock - self.Fock.T.conj()).conj()
        orbgrad *= self.nrr

        return orbgrad

    def _compute_orbhess(self):
        """Diagonal orbital Hessian"""
        orbhess = np.zeros_like(self.Fcore)
        diag_F = np.diag(self.Fock1)
        diag_F2 = np.diag(self.Fock2)
        diag_g1 = np.diag(self.g1)
        diag_grad = np.diag(self.Fock)

        # The VC, VA, AC blocks are based on Theor. Chem. Acc. 97, 88-95 (1997)
        # compute virtual-core block
        orbhess[self.virt, self.core] += 2.0 * (
            diag_F[self.virt, None] - diag_F[None, self.core]
        )

        # compute virtual-active block
        orbhess[self.virt, self.actv] += 2.0 * (
            diag_F[self.virt, None] * diag_g1[None, :] - diag_F2[None, self.actv]
        )

        # compute active-core block
        orbhess[self.actv, self.core] += 2.0 * (
            diag_F[self.actv, None] - diag_F[None, self.core]
        )
        orbhess[self.actv, self.core] += 2.0 * (
            diag_g1[:, None] * diag_F[None, self.core] - diag_F2[self.actv, None]
        )

        # if GAS: compute active-active block [see SI of J. Chem. Phys. 152, 074102 (2020)]
        if self.compute_active_hessian:
            eri_actv = self.get_active_space_ints()
            # A. G^{uu}_{vv}
            Guu_ = np.einsum("uxuy,vvxy->uv", eri_actv, self.g2)
            Guu_ += 2.0 * np.einsum("uuxy,vxvy->uv", eri_actv, self.g2)
            Guu_ += np.diag(self.Fcore)[self.actv, None] * diag_g1[None, :]

            # B. G^{uv}_{vu}
            Guv_ = self.Fcore[self.actv, self.actv] * self.g1.T.conj()
            Guv_ += np.einsum("uxvy,vuxy->uv", eri_actv, self.g2)
            Guv_ += 2.0 * np.einsum("uvxy,vxuy->uv", eri_actv, self.g2)

            # compute diagonal hessian
            orbhess[self.actv, self.actv] = 2.0 * (Guu_ + Guu_.T.conj())
            orbhess[self.actv, self.actv] -= 2.0 * (Guv_ + Guv_.T.conj())
            orbhess[self.actv, self.actv] -= 2.0 * (
                diag_grad[self.actv, None] + diag_grad[None, self.actv]
            )
        orbhess = orbhess * self.nrr

        return orbhess
