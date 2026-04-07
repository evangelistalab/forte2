from dataclasses import dataclass

import numpy as np

from .mp2_base import MP2Base
from forte2.scf import RHF


@dataclass
class RMP2(MP2Base):
    """
    Density-Fitted Møller-Plesset perturbation theory (DF-MP2) method with RHF canonical orbitals.

    Request optional quantities with the fluent helpers inherited from
    :class:`MP2Base`, for example ``RMP2().compute_1rdm().compute_2rdm()``.

    Returns
    -------
    float
        MP2 total energy (E_HF + E_corr).
    """

    def __call__(self, parent_method):
        self.parent_method = parent_method
        if not isinstance(parent_method, RHF):
            raise TypeError("RMP2 requires an RHF reference.")
        return self

    def _reference_label(self) -> str:
        return "RHF"

    def _startup(self):
        super()._startup()
        self.eps = self.parent_method.eps[0].copy()
        self.nocc = self.parent_method.na
        self.nvir = self.parent_method.nuocc

    def _build_df_iaQ(self):
        nocc = self.nocc
        C_occ = self.C[0][:, :nocc]
        C_vir = self.C[0][:, nocc:]

        B_Qia = self.fock_builder.B_tensor_gen_block(C_occ, C_vir)
        assert B_Qia.shape[1] == nocc, f"B occ mismatch: {B_Qia.shape[1]} vs {nocc}"
        assert (
            B_Qia.shape[2] == self.nvir
        ), f"B vir mismatch: {B_Qia.shape[2]} vs {self.nvir}"

        return B_Qia.transpose(1, 2, 0).copy()

    def _build_t2_all(self, B):
        """
        Build all MP2 amplitudes t_{ij}^{ab} and antisymmetrized t̃_{ij}^{ab}.

        Shapes
        ------
        B: (nocc, nvir, naux)
        t2, t2_as: (nocc, nocc, nvir, nvir)

        Energy
        ------
        E_corr = Σ_{ijab} (2(ia|jb) - (ib|ja)) * (ia|jb) / Δ_{ij}^{ab}
            = Σ_{ijab} (2 g_{ij}^{ab} - g_{ij}^{ba}) * t_{ij}^{ab}
        """
        eps_i = self.eps[: self.nocc]
        eps_a = self.eps[self.nocc :]

        nocc, nvir = self.nocc, self.nvir
        t2 = np.empty((nocc, nocc, nvir, nvir)) if self.store_t2 else None
        # antisym only in (a,b): t̃ = 2t - t^{ba}
        t2_as = np.empty_like(t2) if self.store_t2 else None

        E_corr = 0.0
        gijab = np.empty((nvir, nvir))
        tijab = np.empty((nvir, nvir))

        eps_vir = np.ascontiguousarray(eps_a[:, None] + eps_a[None, :])

        for i in range(nocc):
            Bi = B[i]  # (nvir, naux)
            for j in range(nocc):
                Bj = B[j]
                np.dot(Bi, Bj.T, out=gijab)
                denom = eps_i[i] + eps_i[j] - eps_vir
                self._safe_divide(gijab, denom, out=tijab)
                if self.store_t2:
                    t2[i, j] = tijab
                    t2_as[i, j] = 2.0 * tijab - tijab.T

                # energy contribution
                E_corr += np.sum((2.0 * gijab - gijab.T) * tijab)

        return t2, t2_as, E_corr

    def _make_mp2_sf_1rdm_intermediates(self, B):
        """
        Fast unrelaxed spin-summed MP2 1-RDM using DF factors B[i,a,Q].
        No full t2 tensor is formed.

        Returns gamma with:
        gamma1_oo = 2I - (dm1occ + dm1occ.T)
        gamma1_vv =      (dm1vir + dm1vir.T)
        """
        eps_o = self.eps[: self.nocc]
        eps_v = self.eps[self.nocc :]
        nocc, nvir = self.nocc, self.nvir
        nmo = nocc + nvir

        gamma1occ = np.zeros((nocc, nocc))
        gamma1vir = np.zeros((nvir, nvir))

        ea_ab = eps_v[:, None] + eps_v[None, :]  # (a,b)

        for i in range(nocc):
            # g_jab = (ia|jb) for all j,a,b via DF
            # B[i]: (a,Q), B: (j,b,Q)
            g_jab = np.einsum("aQ,jbQ->jab", B[i], B, optimize=True)  # (j,a,b)

            denom = (eps_o[i] + eps_o[:, None, None]) - ea_ab[None, :, :]  # (j,a,b)
            t2i = self._safe_divide(g_jab, denom)

            l2i = t2i  # real case; for complex use t2i.conj()

            # gamma1vir += 2*einsum('jca,jcb->ba', l2i, t2i) - einsum('jca,jbc->ba', l2i, t2i)
            gamma1vir += 2.0 * np.einsum(
                "jca,jcb->ba", l2i, t2i, optimize=True
            ) - np.einsum("jca,jbc->ba", l2i, t2i, optimize=True)

            # gamma1occ += 2*einsum('iab,jab->ij', l2i, t2i) - einsum('iab,jba->ij', l2i, t2i)
            gamma1occ += 2.0 * np.einsum(
                "iab,jab->ij", l2i, t2i, optimize=True
            ) - np.einsum("iab,jba->ij", l2i, t2i, optimize=True)

        gamma1 = np.zeros((nmo, nmo))
        gamma1[:nocc, :nocc] = 2.0 * np.eye(nocc)
        gamma1[:nocc, :nocc] += -(gamma1occ + gamma1occ.T)
        gamma1[nocc:, nocc:] += gamma1vir + gamma1vir.T
        return gamma1

    def _make_mp2_sf_2rdm(self, t2, gamma1, store_debug=False):
        nocc, nvir = self.nocc, self.nvir
        nmo = nocc + nvir

        gamma2 = np.zeros((nmo, nmo, nmo, nmo), dtype=t2.dtype)

        o = np.arange(nocc)
        v = np.arange(nocc, nmo)

        # -------------------------
        # (1) OVOV / VOVO from t2
        # -------------------------
        # Build dovov_iajb with shape (i,a,j,b)
        # dovov = (2*t2[i,j,a,b] - t2[i,j,b,a]) * 2, but rearranged to (i,a,j,b)
        dovov_iajb = (2.0 * t2.transpose(0, 2, 1, 3) - t2.transpose(0, 3, 1, 2)) * 2.0
        # Fill gamma2[o, v, o, v]
        gamma2[np.ix_(o, v, o, v)] = dovov_iajb

        # For real case: gamma2[v,i,v,j] = dovov[a,j,b] with axes (a,b,j)
        gamma2[np.ix_(v, o, v, o)] = dovov_iajb.transpose(1, 0, 3, 2)

        # -------------------------
        # (2) Addback from gamma1 + constants
        # -------------------------
        gamma1_work = gamma1.copy()
        gamma1_work[o, o] -= 2.0  # subtract 2 on occupied diagonal only
        gamma1T = gamma1_work.T

        o = np.arange(nocc)

        gamma2[o, o, :, :] += 2.0 * gamma1T[None, :, :]  # (nocc,nmo,nmo) += (1,nmo,nmo)
        gamma2[:, :, o, o] += 2.0 * gamma1T[:, :, None]  # (nmo,nmo,nocc) += (nmo,nmo,1)
        gamma2[:, o, o, :] -= gamma1T[:, None, :]  # (nmo,nocc,nmo) -= (nmo,1,nmo)
        gamma2[o, :, :, o] -= gamma1_work[None, :, :]  # (nocc,nmo,nmo) -= (1,nmo,nmo)

        # constants:
        # gamma2[i,i,j,j] += 4
        gamma2[o[:, None], o[:, None], o[None, :], o[None, :]] += 4.0
        # gamma2[i,j,j,i] -= 2
        gamma2[o[:, None], o[None, :], o[None, :], o[:, None]] -= 2.0

        if store_debug:
            # Store only the small blocks, not full nmo^4 arrays
            self._dm2_ovov = dovov_iajb.copy()  # (i,a,j,b)
            # You can also store gamma2_add restricted to OO/OO etc if needed

        return gamma2

    def _make_mp2_sf_2cumulants(self, gamma1, gamma2):
        """
        gamma1[q,p] = <p† q>
        gamma2[p,q,r,s] = < p† r† s q >

        Disconnected/HF-like part:
        gamma2^(0)[p,q,r,s] = gamma1[q,p]*gamma1[s,r] - 1/2 * gamma1[q,r]*gamma1[s,p]
        """
        term1 = np.einsum("qp,sr->pqrs", gamma1, gamma1, optimize=True)
        term2 = np.einsum("qr,sp->pqrs", gamma1, gamma1, optimize=True)
        gamma2_0 = term1 - 0.5 * term2

        return gamma2 - gamma2_0

    def make_1rdm(self):
        return self._make_mp2_sf_1rdm_intermediates(self.B_iaQ)

    def make_2rdm(self, gamma1=None):
        if not self.store_t2:
            raise ValueError("t2 amplitudes were not stored. Cannot compute 2-RDM.")
        if gamma1 is None:
            gamma1 = self.make_1rdm()
        return self._make_mp2_sf_2rdm(self.t2, gamma1)

    def make_2cumulant(self, gamma1=None, gamma2=None):
        if not self.store_t2:
            raise ValueError(
                "t2 amplitudes were not stored. Cannot compute 2-cumulant."
            )
        if gamma1 is None:
            gamma1 = self.make_1rdm()
        if gamma2 is None:
            gamma2 = self.make_2rdm(gamma1)
        return self._make_mp2_sf_2cumulants(gamma1, gamma2)

    def make_cumulants(self):
        gamma1 = self.make_1rdm()
        gamma2 = self.make_2rdm(gamma1)
        lambda2 = self.make_2cumulant(gamma1, gamma2)
        return gamma1, gamma2, lambda2

    def energy_given_rdms(self, Ecore, H, V, gamma1, gamma2):
        """
        Computes the MP2 energy from contracting rdms
        """
        e1 = np.einsum("pq,qp->", H, gamma1, optimize=True)
        e2 = 0.5 * np.einsum("pqrs,prqs", V, gamma2, optimize=True)

        return Ecore + e1 + e2
