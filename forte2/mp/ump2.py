from dataclasses import dataclass
import time

import numpy as np

from .mp2_base import MP2Base
from forte2.base_classes import SystemMixin, MOsMixin
from forte2.scf import UHF


@dataclass
class UMP2(MP2Base):
    """
    Density-Fitted Møller-Plesset perturbation theory (DF-MP2) method with UHF canonical orbitals.

    Request optional quantities with the fluent helpers inherited from
    :class:`MP2Base`, for example ``UMP2().compute_1rdm().compute_2rdm()``.

    Returns
    -------
    float
        MP2 total energy (E_HF + E_corr).
    """

    def __call__(self, parent_method):
        self.parent_method = parent_method
        if not isinstance(parent_method, UHF):
            raise TypeError("UMP2 requires an UHF reference.")
        return self

    def _reference_label(self) -> str:
        return "UHF"

    def _t2_norm(self):
        if any(getattr(self, attr, None) is None for attr in ("t2_a", "t2_b", "t2_ab")):
            return None
        return (
            np.linalg.norm(self.t2_a)
            + np.linalg.norm(self.t2_b)
            + np.linalg.norm(self.t2_ab)
        )

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        # Copy system + MO information
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        # MO reordering (contiguous correlated space)
        self.Ca = self.C[0].copy()
        self.Cb = self.C[1].copy()
        self.eps_a = self.parent_method.eps[0].copy()
        self.eps_b = self.parent_method.eps[1].copy()

        # occupations
        self.naocc = self.parent_method.na
        self.nbocc = self.parent_method.nb

        # total MOs
        self.nmo = self.parent_method.nmo

        # virtuals
        self.navir = self.nmo - self.naocc
        self.nbvir = self.nmo - self.nbocc

        self.fock_builder = self.system.fock_builder

    def run(self):
        t0 = time.monotonic()
        mem0 = self._memory_snapshot()

        self._startup()
        need_t2 = self._needs_t2_storage()

        self.B_iaQ = self._build_df_iaQ()

        (self.t2_a, self.t2_b, self.t2_ab, self.E_corr) = self._build_t2_all(
            self.B_iaQ, store_t2=need_t2
        )

        self.E_total = self.parent_method.E + self.E_corr
        self._initialize_rdm_outputs()
        self.gamma2_aa = None
        self.gamma2_ab = None
        self.gamma2_bb = None
        need_gamma1 = (
            self._compute_1rdm
            or self._compute_1rdm_ao
            or self._compute_2rdm
            or self._compute_cumulants
        )
        # MP2Base assumes spin-restricted tensors. UHF overrides all RDM builders.
        if need_gamma1:
            self.make_mp2_sf_1rdm_intermediates(self.B_iaQ)

            if self._compute_1rdm_ao:
                self.gamma1_sf_ao = self.gamma1_mo_to_ao(self.gamma1_sf)

        if self._compute_2rdm or self._compute_cumulants:
            self.gamma2_sf = self.make_mp2_sf_2rdm()

        if self._compute_cumulants:
            self.lambda2_sf = self.make_mp2_sf_2cumulants(
                self.gamma1_sf, self.gamma2_sf
            )

        self.executed = True
        self._log_completion(time.monotonic() - t0, self._t2_norm(), mem0)
        return self.E_total

    def _build_df_iaQ(self):
        """
        Build spin-resolved 3-index integrals (ia|Q) using density fitting.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Alpha and beta DF tensors with shapes
            ``(naocc, navir, naux)`` and ``(nbocc, nbvir, naux)``.
        """
        Ca_occ = self.Ca[:, : self.naocc]
        Ca_vir = self.Ca[:, self.naocc :]
        Cb_occ = self.Cb[:, : self.nbocc]
        Cb_vir = self.Cb[:, self.nbocc :]

        Ba_Qia = self.fock_builder.B_tensor_gen_block(Ca_occ, Ca_vir)
        Bb_Qia = self.fock_builder.B_tensor_gen_block(Cb_occ, Cb_vir)

        assert Ba_Qia.shape[1] == self.naocc
        assert Ba_Qia.shape[2] == self.navir
        assert Bb_Qia.shape[1] == self.nbocc
        assert Bb_Qia.shape[2] == self.nbvir

        return (
            Ba_Qia.transpose(1, 2, 0).copy(),
            Bb_Qia.transpose(1, 2, 0).copy(),
        )

    def _build_t2_all(self, B, store_t2=True):
        Ba, Bb = B

        eps_a_i = self.eps_a[: self.naocc]
        eps_a_a = self.eps_a[self.naocc :]
        eps_b_i = self.eps_b[: self.nbocc]
        eps_b_a = self.eps_b[self.nbocc :]

        naocc, navir = self.naocc, self.navir
        nbocc, nbvir = self.nbocc, self.nbvir

        # allocate
        t2_a = np.zeros((naocc, naocc, navir, navir)) if store_t2 else None
        t2_b = np.zeros((nbocc, nbocc, nbvir, nbvir)) if store_t2 else None
        t2_ab = np.zeros((naocc, nbocc, navir, nbvir)) if store_t2 else None

        E_corr = 0.0
        gijab_aa = np.empty((navir, navir))
        tijab_aa = np.empty((navir, navir))
        gijab_bb = np.empty((nbvir, nbvir))
        tijab_bb = np.empty((nbvir, nbvir))
        gijab_ab = np.empty((navir, nbvir))
        tijab_ab = np.empty((navir, nbvir))
        eps_aa_vv = np.ascontiguousarray(eps_a_a[:, None] + eps_a_a[None, :])
        eps_bb_vv = np.ascontiguousarray(eps_b_a[:, None] + eps_b_a[None, :])
        eps_ab_vv = np.ascontiguousarray(eps_a_a[:, None] + eps_b_a[None, :])

        # =========================
        # ALPHA-ALPHA
        # =========================
        for i in range(naocc):
            Bi = Ba[i]  # (navir, naux)
            for j in range(naocc):
                Bj = Ba[j]

                np.dot(Bi, Bj.T, out=gijab_aa)
                g_as = gijab_aa - gijab_aa.T
                denom = eps_a_i[i] + eps_a_i[j] - eps_aa_vv
                self._safe_divide(g_as, denom, out=tijab_aa)

                if store_t2:
                    t2_a[i, j] = tijab_aa

                # energy (same-spin → 1/4 factor)
                E_corr += 0.25 * np.sum(g_as * tijab_aa)

        # =========================
        # BETA-BETA
        # =========================
        for i in range(nbocc):
            Bi = Bb[i]
            for j in range(nbocc):
                Bj = Bb[j]

                np.dot(Bi, Bj.T, out=gijab_bb)
                g_as = gijab_bb - gijab_bb.T
                denom = eps_b_i[i] + eps_b_i[j] - eps_bb_vv
                self._safe_divide(g_as, denom, out=tijab_bb)

                if store_t2:
                    t2_b[i, j] = tijab_bb

                E_corr += 0.25 * np.sum(g_as * tijab_bb)

        # =========================
        # ALPHA-BETA
        # =========================
        for i in range(naocc):
            Bi = Ba[i]
            for j in range(nbocc):
                Bj = Bb[j]

                np.dot(Bi, Bj.T, out=gijab_ab)
                denom = eps_a_i[i] + eps_b_i[j] - eps_ab_vv
                self._safe_divide(gijab_ab, denom, out=tijab_ab)

                if store_t2:
                    t2_ab[i, j] = tijab_ab

                # opposite-spin → no 1/4 factor
                E_corr += np.sum(gijab_ab * tijab_ab)

        return t2_a, t2_b, t2_ab, E_corr

    def make_mp2_sf_1rdm_intermediates(self, B):
        # Rebuild amplitudes locally when they were not requested for storage.
        if all(getattr(self, attr, None) is not None for attr in ("t2_a", "t2_b", "t2_ab")):
            t2_a = self.t2_a
            t2_b = self.t2_b
            t2_ab = self.t2_ab
        else:
            t2_a, t2_b, t2_ab, _ = self._build_t2_all(B, store_t2=True)

        naocc = self.naocc
        nbocc = self.nbocc

        # =========================
        # ALPHA BLOCKS
        # =========================

        # occupied-occupied
        doo_a = -0.5 * (
            np.einsum("imef,jmef->ij", t2_a, t2_a, optimize=True)
            + np.einsum("iMef,jMef->ij", t2_ab, t2_ab, optimize=True)
        )

        # virtual-virtual
        dvv_a = 0.5 * (
            np.einsum("mnae,mnbe->ab", t2_a, t2_a, optimize=True)
            + np.einsum("mNae,mNbe->ab", t2_ab, t2_ab, optimize=True)
        )

        # =========================
        # BETA BLOCKS
        # =========================

        doo_b = -0.5 * (
            np.einsum("imef,jmef->ij", t2_b, t2_b, optimize=True)
            + np.einsum("mief,mjef->ij", t2_ab, t2_ab, optimize=True)
        )

        dvv_b = 0.5 * (
            np.einsum("mnae,mnbe->ab", t2_b, t2_b, optimize=True)
            + np.einsum("mnea,mneb->ab", t2_ab, t2_ab, optimize=True)
        )

        # =========================
        # BUILD DENSITY MATRICES
        # =========================

        gamma1_a = np.zeros((self.nmo, self.nmo), dtype=t2_a.dtype)
        gamma1_b = np.zeros((self.nmo, self.nmo), dtype=t2_b.dtype)

        # reference occupations
        gamma1_a[:naocc, :naocc] = np.eye(naocc)
        gamma1_b[:nbocc, :nbocc] = np.eye(nbocc)

        # add MP2 corrections
        gamma1_a[:naocc, :naocc] += doo_a
        gamma1_a[naocc:, naocc:] += dvv_a

        gamma1_b[:nbocc, :nbocc] += doo_b
        gamma1_b[nbocc:, nbocc:] += dvv_b

        # enforce Hermiticity
        gamma1_a = 0.5 * (gamma1_a + gamma1_a.T)
        gamma1_b = 0.5 * (gamma1_b + gamma1_b.T)

        # spin-free
        gamma1_sf = gamma1_a + gamma1_b
        gamma1_sf = 0.5 * (gamma1_sf + gamma1_sf.T)

        self.gamma1_a = gamma1_a
        self.gamma1_b = gamma1_b
        self.gamma1_sf = gamma1_sf

        return gamma1_sf

    def gamma1_mo_to_ao(self):
        return self.Ca @ self.gamma1_a @ self.Ca.T + self.Cb @ self.gamma1_b @ self.Cb.T

    def make_mp2_sf_2rdm(self):
        t2_a = self.t2_a
        t2_b = self.t2_b
        t2_ab = self.t2_ab

        nmo = self.nmo
        naocc, navir = self.naocc, self.navir
        nbocc, nbvir = self.nbocc, self.nbvir

        oa = np.arange(naocc)
        va = np.arange(naocc, nmo)
        ob = np.arange(nbocc)
        vb = np.arange(nbocc, nmo)

        # =========================
        # Allocate spin blocks
        # =========================
        common_dtype = np.result_type(t2_a, t2_b, t2_ab)
        dm2_aa = np.zeros((nmo, nmo, nmo, nmo), dtype=common_dtype)
        dm2_bb = np.zeros((nmo, nmo, nmo, nmo), dtype=common_dtype)
        dm2_ab = np.zeros((nmo, nmo, nmo, nmo), dtype=common_dtype)

        # =========================
        # SAME-SPIN (αα)
        # =========================
        # (ijab)
        dm2_aa[np.ix_(oa, oa, va, va)] += t2_a
        dm2_aa[np.ix_(va, va, oa, oa)] += t2_a.transpose(2, 3, 0, 1)

        # antisymmetry permutations
        dm2_aa[np.ix_(oa, oa, va, va)] -= t2_a.transpose(0, 1, 3, 2)
        dm2_aa[np.ix_(va, va, oa, oa)] -= t2_a.transpose(3, 2, 0, 1)

        # =========================
        # SAME-SPIN (ββ)
        # =========================
        dm2_bb[np.ix_(ob, ob, vb, vb)] += t2_b
        dm2_bb[np.ix_(vb, vb, ob, ob)] += t2_b.transpose(2, 3, 0, 1)

        dm2_bb[np.ix_(ob, ob, vb, vb)] -= t2_b.transpose(0, 1, 3, 2)
        dm2_bb[np.ix_(vb, vb, ob, ob)] -= t2_b.transpose(3, 2, 0, 1)

        # =========================
        # OPPOSITE-SPIN (αβ)
        # =========================
        dm2_ab[np.ix_(oa, ob, va, vb)] += t2_ab
        dm2_ab[np.ix_(va, vb, oa, ob)] += t2_ab.transpose(2, 3, 0, 1)

        # =========================
        # Add reference (HF) part
        # =========================
        gamma1_a = self.gamma1_a
        gamma1_b = self.gamma1_b

        # αα
        dm2_aa += np.einsum("pr,qs->pqrs", gamma1_a, gamma1_a) - np.einsum(
            "ps,qr->pqrs", gamma1_a, gamma1_a
        )

        # ββ
        dm2_bb += np.einsum("pr,qs->pqrs", gamma1_b, gamma1_b) - np.einsum(
            "ps,qr->pqrs", gamma1_b, gamma1_b
        )

        # αβ
        dm2_ab += np.einsum("pr,qs->pqrs", gamma1_a, gamma1_b)

        # =========================
        # Spin-free assembly
        # =========================
        self.gamma2_aa = dm2_aa
        self.gamma2_ab = dm2_ab
        self.gamma2_bb = dm2_bb
        gamma2_sf = dm2_aa + dm2_bb + dm2_ab + dm2_ab.transpose(1, 0, 3, 2)

        # =========================
        # Enforce symmetries
        # =========================
        gamma2_sf = 0.5 * (gamma2_sf + gamma2_sf.transpose(1, 0, 3, 2))
        gamma2_sf = 0.5 * (gamma2_sf + gamma2_sf.transpose(2, 3, 0, 1))

        return gamma2_sf

    def make_mp2_sf_2cumulants(self, gamma1, gamma2):
        term1 = np.einsum("pr,qs->pqrs", gamma1, gamma1)
        term2 = np.einsum("ps,qr->pqrs", gamma1, gamma1)
        dm2_0 = term1 - 0.5 * term2
        return gamma2 - dm2_0

    def mp2_E_given_rdms(self, Ecore, H, V, gamma1_a, gamma1_b, gamma2_a, gamma2_b):
        """
        Computes mp2 Energy from comtracting rdms
        """
        e1 = np.einsum("pq,qp->", H, gamma1_a + gamma1_b, optimize=True)

        gamma2_aa = gamma2_a
        gamma2_ab = None
        gamma2_bb = gamma2_b

        if isinstance(gamma2_a, tuple):
            if len(gamma2_a) == 3:
                gamma2_aa, gamma2_ab, gamma2_bb = gamma2_a
            elif len(gamma2_a) == 2:
                gamma2_aa, gamma2_ab = gamma2_a

        if isinstance(gamma2_b, tuple):
            if len(gamma2_b) == 3:
                gamma2_aa, gamma2_ab, gamma2_bb = gamma2_b
            elif len(gamma2_b) == 2:
                gamma2_ab, gamma2_bb = gamma2_b

        e2 = 0.5 * np.einsum("pqrs,prqs", V, gamma2_aa + gamma2_bb, optimize=True)
        if gamma2_ab is not None:
            e2 += np.einsum("pqrs,prqs", V, gamma2_ab, optimize=True)

        return Ecore + e1 + e2
