import numpy as np
import scipy as sp

from forte2.jkbuilder import FockBuilder


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
        # Mask applied when screening the gradient and Hessian. It is the union of
        # every rotation block, which differs from `nrr` only when a second block
        # (electronic-positronic) is driven alongside the electronic one.
        self.rotation_mask = nrr
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

    def compute_orbital_lagrangian(self):
        r"""
        Return the symmetric CASSCF orbital Lagrangian matrix.

        The orbital optimizer forms the matrix :math:`A_{pq}` whose
        antisymmetric part is the orbital gradient,

        .. math::
            g_{pq} = 2(A_{pq} - A_{qp}).

        At a fully optimized state-specific CASSCF solution, the nonredundant
        antisymmetric part vanishes.  The symmetric part of :math:`A` is the
        molecular-orbital energy-weighted density used in the Pulay overlap
        derivative contribution,

        .. math::
            W^{S}_{\mu\nu}
            =
            C_{\mu p}
            \frac{1}{2}(A_{pq}+A_{qp})
            C_{\nu q}.

        Returns
        -------
        np.ndarray
            Symmetric orbital Lagrangian in the current MO basis.
        """
        self._compute_orbgrad()
        return 0.5 * (self.A_pq + self.A_pq.T.conj())

    def reference_energy(self):
        """Return the energy at the current orbitals."""
        return self._compute_reference_energy()

    def rotate(self, mask, dx):
        """Apply an incremental rotation confined to one block of the mask.

        `evaluate` tracks a cumulative rotation vector against a single mask.
        This takes the increment directly instead, so several disjoint blocks can
        drive the same optimizer without sharing a parameterization.
        """
        if dx.size == 0 or np.max(np.abs(dx)) < 1e-12:
            return
        nmo = self.C.shape[1]
        X = np.zeros((nmo, nmo), dtype=self.C.dtype)
        X[mask] = dx
        X -= X.conj().T
        self.U = self.U @ sp.linalg.expm(X)
        self.C = self.C0 @ self.U
        self.Cgen = self.C
        self.Ccore = self.C[:, self.core]
        self.Cact = self.C[:, self.actv]
        self._compute_Fcore()
        self.get_eri_gaaa()

    def _update_orbitals(self, R):
        dR = R - self.R
        if np.max(np.abs(dR)) < 1e-12:
            # no change in orbitals, skip the update
            return False
        self.R += dR
        self.U = self.U @ self._expm(dR)

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
        # The generator lives in the MO basis, which is smaller than the AO basis
        # whenever the metric is truncated, and half the size for four-component
        # spinors.
        nmo = self.C.shape[1]
        R = np.zeros((nmo, nmo), dtype=self.C.dtype)
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

    @staticmethod
    def _transform_ao_operator(operator, C):
        return np.einsum(
            "mp,mn,nq->pq",
            C.conj(),
            operator,
            C,
            optimize=True,
        )

    def _compute_Fcore(self):
        # Compute the core Fock matrix [eq (3)], also return the core energy
        Fcore_ao = self.fock_builder.build_core_fock(self.Ccore, hcore=self.hcore)
        self.Fcore = self._transform_ao_operator(Fcore_ao, self.Cgen)

        core_factor = self.fock_builder.core_energy_factor
        self.Ecore = core_factor * np.trace(
            self._transform_ao_operator(
                self.hcore + Fcore_ao,
                self.Ccore,
            )
        )

    def _compute_Fact(self):
        # [eq (13)]
        Fact_ao = self.fock_builder.build_active_fock(self.Cact, self.g1)
        self.Fact = self._transform_ao_operator(Fact_ao, self.Cgen)

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
        orbgrad *= self.rotation_mask

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
        orbhess *= self.rotation_mask

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

    def compute_orbital_lagrangian(self):
        """Return the Hermitian two-component CASSCF orbital Lagrangian."""
        self._compute_orbgrad()
        # RelOrbOptimizer stores its generalized Fock matrix with the
        # Lagrangian indices transposed relative to the AO transformation.
        return 0.5 * (self.Fock + self.Fock.T.conj()).T

    def _compute_reference_energy(self):
        energy = self.Ecore + self.e_nuc
        energy += np.einsum("uv,uv->", self.Fcore[self.actv, self.actv], self.g1)
        energy += 0.5 * np.einsum("tvuw,tuvw->", self.get_active_space_ints(), self.g2)
        return energy

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
        orbgrad *= self.rotation_mask

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
        orbhess = orbhess * self.rotation_mask

        return orbhess


class RotationSubspace:
    r"""
    One block of an orbital rotation space, presented as an L-BFGS objective.

    Pass ``maximize=True`` for the electronic-positronic block of a
    four-component optimization. The Dirac-CASSCF energy is a minimum with
    respect to rotations among electronic orbitals but a *maximum* with respect
    to rotations between electronic and positronic ones, because the Hessian
    there carries the :math:`-2c^2` energy denominator. Negating the objective
    lets an ordinary minimizer carry out that maximization, which keeps L-BFGS
    inside the positive-curvature regime that both its diagonal preconditioner
    and its curvature guard assume.

    Driving the two blocks as separate objectives also means they never share an
    L-BFGS history, so a search direction cannot pick up a wrong-signed component
    from the other block. This follows Bates and Shiozaki, J. Chem. Phys. 142,
    064112 (2015), who alternate the two optimizations for the same reason.

    Parameters
    ----------
    optimizer : OrbOptimizer
        The optimizer supplying the energy, gradient and diagonal Hessian.
    mask : NDArray
        Boolean mask selecting this block out of the full rotation matrix.
    maximize : bool, optional, default=False
        Whether to maximize rather than minimize over this block.
    """

    def __init__(self, optimizer, mask, maximize=False):
        self.optimizer = optimizer
        self.mask = mask
        self.sign = -1.0 if maximize else 1.0
        self.nrot = int(mask.sum())
        # `R` is handed to the optimizer, which mutates it in place, so the
        # rotation already applied has to be tracked separately.
        self.R = np.zeros(self.nrot, dtype=optimizer.C.dtype)
        self._applied = np.zeros(self.nrot, dtype=optimizer.C.dtype)

    def evaluate(self, x):
        self.optimizer.rotate(self.mask, x - self._applied)
        self._applied[:] = x
        return self.sign * self.optimizer.reference_energy()

    def gradient(self, x):
        return self.sign * self.optimizer._compute_orbgrad()[self.mask]

    def hess_diag(self, x):
        return self.sign * self.optimizer._compute_orbhess()[self.mask]
