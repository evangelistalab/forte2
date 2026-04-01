from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

import numpy as np

from forte2.base_classes import SystemMixin, MOsMixin
from forte2.scf import RHF, ROHF, UHF
from forte2.helpers import logger


@dataclass
class MP2Base(SystemMixin, MOsMixin, ABC):
    """Base class for density-fitted MP2 methods. Not meant to be used directly.

    Parameters
    ----------
    compute_1rdm : bool
        If True, build the spin-free 1-RDM (unrelaxed MP2).
    compute_1rdm_ao : bool
        If True, build the spin-free 1-RDM in AO basis.
    compute_2rdm : bool
        If True, build the spin-free 2-RDM (potentially large).
    compute_cumulants : bool
        If True, build the 2-body cumulant (and 1-body hole RDM if needed).
        Usually implies compute_2rdm.

    Attributes
    ----------
    parent_method : RHF or ROHF
        Reference wavefunction object providing orbitals, orbital energies,
        occupation numbers, and Fock builder.
    C : ndarray
        Molecular orbital coefficient matrix in AO basis.
    eps : ndarray
        Orbital energies in the working (possibly semicanonical) basis.
    nocc : int
        Number of correlated occupied orbitals.
    nvir : int
        Number of correlated virtual orbitals.
    B_iaQ : ndarray
        Density-fitted three-index integrals (ia|Q).
    t2 : ndarray
        MP2 double-excitation amplitudes.
    E_corr : float
        MP2 correlation energy.
    E_total : float
        Total energy (E_reference + E_corr).
    gamma1_sf : ndarray or None
        Spin-free one-particle reduced density matrix (if requested).
    gamma2_sf : ndarray or None
        Spin-free two-particle reduced density matrix (if requested).
    lambda2_sf : ndarray or None
        Spin-free two-body cumulant (if requested).
    executed : bool
        Whether the MP2 calculation has been executed.

    NOTE:
    -----
    This base class assumes spin-restricted tensors:
        B: (nocc, nvir, naux)
        t2: (nocc, nocc, nvir, nvir)

    UHF overrides:
        - _build_df_iaQ
        - _build_t2_all
        - RDM builders

    and does NOT use base-class tensor conventions.
    """

    compute_1rdm: bool = False
    compute_1rdm_ao: bool = False
    compute_2rdm: bool = False
    compute_cumulants: bool = False
    executed: bool = field(default=False, init=False)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        if not isinstance(parent_method, (RHF, ROHF, UHF)):
            raise TypeError("MP2 requires an RHF or ROHF reference.")
        return self

    def run(self):
        t0 = time.monotonic()

        self._startup()

        self.B_iaQ = self._build_df_iaQ()

        self.t2, self.t2_as, self.E_corr = self._build_t2_all(self.B_iaQ)

        self.E_total = self.parent_method.E + self.E_corr

        self._postprocess_rdms()

        # Print information
        if isinstance(self.parent_method, RHF):
            logger.log_info1("RHF-MP2 calculation completed.")
        elif isinstance(self.parent_method, ROHF):
            logger.log_info1("ROHF-MP2 calculation completed.")
        elif isinstance(self.parent_method, UHF):
            logger.log_info1("UHF-MP2 calculation completed.")

        logger.log_info1(f"E(corr) = {self.E_corr:.13f} Eh")
        logger.log_info1(f"E(total) = {self.E_total:.13f} Eh")
        logger.log_info1(f"||t2|| = {np.linalg.norm(self.t2)}")

        self.executed = True
        dt = time.monotonic() - t0
        logger.log_info1(f"Time = {dt:.3f} s")
        return self.E_total

    def _postprocess_rdms(self):
        self.gamma1_sf = None
        self.gamma2_sf = None
        self.lambda2_sf = None

        if self.compute_1rdm:
            self.gamma1_sf = self.make_mp2_sf_1rdm_intermediates(self.B_iaQ)

            if self.compute_1rdm_ao:
                self.gamma1_sf_ao = self.gamma1_mo_to_ao(self.gamma1_sf)

        if self.compute_2rdm or self.compute_cumulants:
            self.gamma2_sf = self.make_mp2_sf_2rdm(self.t2, self.gamma1_sf)

        if self.compute_cumulants:
            self.lambda2_sf = self.make_mp2_sf_2cumulants(
                self.gamma1_sf, self.gamma2_sf
            )
        if isinstance(self.parent_method, UHF):
            raise RuntimeError(
                "MP2Base._postprocess_rdms should not be used for UHF. "
                "Use UHFMP2 workflow instead."
            )

    def make_mp2_sf_1rdm_intermediates(self, B: np.ndarray) -> np.ndarray:
        """
        Fast unrelaxed spin-summed MP2 1-RDM (PySCF-style) using DF factors B[i,a,Q].
        No full t2 tensor is formed.

        Returns gamma with:
        gamma_oo = 2I - (dm1occ + dm1occ.T)
        gamma_vv =      (dm1vir + dm1vir.T)
        where dm1occ, dm1vir match PySCF's _gamma1_intermediates (up to symmetry handling).
        """
        eps_o = self.eps[: self.nocc]
        eps_v = self.eps[self.nocc :]
        nocc, nvir = self.nocc, self.nvir
        nmo = nocc + nvir

        dm1occ = np.zeros((nocc, nocc))
        dm1vir = np.zeros((nvir, nvir))

        ea_ab = eps_v[:, None] + eps_v[None, :]  # (a,b)

        for i in range(nocc):
            # g_jab = (ia|jb) for all j,a,b via DF
            # B[i]: (a,Q), B: (j,b,Q)
            g_jab = np.einsum("aQ,jbQ->jab", B[i], B, optimize=True)  # (j,a,b)

            denom = (eps_o[i] + eps_o[:, None, None]) - ea_ab[None, :, :]  # (j,a,b)
            t2i = self._safe_divide(g_jab, denom)

            l2i = t2i  # real case; for complex use t2i.conj()

            # dm1vir += 2*einsum('jca,jcb->ba', l2i, t2i) - einsum('jca,jbc->ba', l2i, t2i)
            dm1vir += 2.0 * np.einsum(
                "jca,jcb->ba", l2i, t2i, optimize=True
            ) - np.einsum("jca,jbc->ba", l2i, t2i, optimize=True)

            # dm1occ += 2*einsum('iab,jab->ij', l2i, t2i) - einsum('iab,jba->ij', l2i, t2i)
            dm1occ += 2.0 * np.einsum(
                "iab,jab->ij", l2i, t2i, optimize=True
            ) - np.einsum("iab,jba->ij", l2i, t2i, optimize=True)

        gamma = np.zeros((nmo, nmo))
        gamma[:nocc, :nocc] = 2.0 * np.eye(nocc)
        gamma[:nocc, :nocc] += -(dm1occ + dm1occ.T)
        gamma[nocc:, nocc:] += dm1vir + dm1vir.T
        return gamma

    def gamma1_mo_to_ao(self, gamma1_mo: np.ndarray) -> np.ndarray:
        """
        Transform spin-free 1-RDM from MO basis to AO basis.

        Convention:
        MO coefficients C are AO->MO: phi_p = sum_mu C[mu,p] chi_mu
        gamma_AO = C gamma_MO C^T  (real orbitals; use C.conj() for complex)
        """
        C = self.C  # shape (nao, nmo)
        return C @ gamma1_mo @ C.T

    def make_mp2_sf_2rdm(self, t2, dm1, store_debug=False):
        nocc, nvir = self.nocc, self.nvir
        nmo = nocc + nvir

        dm2 = np.zeros((nmo, nmo, nmo, nmo), dtype=t2.dtype)

        o = np.arange(nocc)
        v = np.arange(nocc, nmo)

        # -------------------------
        # (1) OVOV / VOVO from t2
        # -------------------------
        # Build dovov_iajb with shape (i,a,j,b)
        # dovov = (2*t2[i,j,a,b] - t2[i,j,b,a]) * 2, but rearranged to (i,a,j,b)
        dovov_iajb = (2.0 * t2.transpose(0, 2, 1, 3) - t2.transpose(0, 3, 1, 2)) * 2.0
        # Fill dm2[o, v, o, v]
        dm2[np.ix_(o, v, o, v)] = dovov_iajb

        # For real case: dm2[v,i,v,j] = dovov[a,j,b] with axes (a,b,j)
        dm2[np.ix_(v, o, v, o)] = dovov_iajb.transpose(1, 0, 3, 2)

        # -------------------------
        # (2) Addback from dm1 + constants
        # -------------------------
        dm1_work = dm1.copy()
        dm1_work[o, o] -= 2.0  # subtract 2 on occupied diagonal only
        dm1T = dm1_work.T

        o = np.arange(nocc)

        dm2[o, o, :, :] += 2.0 * dm1T[None, :, :]  # (nocc,nmo,nmo) += (1,nmo,nmo)
        dm2[:, :, o, o] += 2.0 * dm1T[:, :, None]  # (nmo,nmo,nocc) += (nmo,nmo,1)
        dm2[:, o, o, :] -= dm1T[:, None, :]  # (nmo,nocc,nmo) -= (nmo,1,nmo)
        dm2[o, :, :, o] -= dm1_work[None, :, :]  # (nocc,nmo,nmo) -= (1,nmo,nmo)

        # constants:
        # dm2[i,i,j,j] += 4
        dm2[o[:, None], o[:, None], o[None, :], o[None, :]] += 4.0
        # dm2[i,j,j,i] -= 2
        dm2[o[:, None], o[None, :], o[None, :], o[:, None]] -= 2.0

        if store_debug:
            # Store only the small blocks, not full nmo^4 arrays
            self._dm2_ovov = dovov_iajb.copy()  # (i,a,j,b)
            # You can also store dm2_add restricted to OO/OO etc if needed

        return dm2

    def make_mp2_sf_2cumulants(self, gamma1, gamma2):
        """
        PySCF conventions:
        dm1[q,p] = <p† q>
        dm2[p,q,r,s] = < p† r† s q >

        Disconnected/HF-like part:
        dm2^(0)[p,q,r,s] = dm1[q,p]*dm1[s,r] - 1/2 * dm1[q,r]*dm1[s,p]
        """
        dm1 = gamma1
        dm2 = gamma2

        term1 = np.einsum("qp,sr->pqrs", dm1, dm1, optimize=True)
        term2 = np.einsum("qr,sp->pqrs", dm1, dm1, optimize=True)
        dm2_0 = term1 - 0.5 * term2

        return dm2 - dm2_0

    def mp2_E_given_rdms(self, Ecore, H, V, gamma1, gamma2):
        """
        Computes mp2 Energy from comtracting rdms
        """
        e1 = np.einsum("pq,qp->", H, gamma1, optimize=True)
        e2 = 0.5 * np.einsum("pqrs,prqs", V, gamma2, optimize=True)

        return Ecore + e1 + e2

    def _safe_divide(self, num, denom, tiny=1e-12, label="MP2 denom"):
        mask = np.abs(denom) < tiny
        n_bad = np.count_nonzero(mask)
        if n_bad:
            logger.log_warning(f"{label}: {n_bad} / {denom.size} elements < {tiny:g}")
        return num / np.where(mask, np.inf, denom)

    @abstractmethod
    def _build_df_iaQ(self): ...

    @abstractmethod
    def _startup(self): ...

    @abstractmethod
    def _build_t2_all(self, B): ...


@dataclass
class RHFMP2(MP2Base):
    """
    Density-Fitted Møller–Plesset perturbation theory (DF-MP2) method with RHF canonical orbitals.

    Parameters
    ----------
    compute_1rdm
        If True, build the spin-free 1-RDM (unrelaxed MP2).
    compute_1rdm_ao
        If True, build the spin-free 1-RDM in AO basis.
    compute_2rdm
        If True, build the spin-free 2-RDM (potentially large).
    compute_cumulants
        If True, build 2-body cumulant (and 1-body hole RDM if needed).
    Returns
    -------
    float
        MP2 total energy (E_HF + E_corr).
    """

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        # Copy system + MO information
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        # MO reordering (contiguous correlated space)
        self.C = self.C[0].copy()
        self.eps = self.parent_method.eps[0].copy()

        self.nocc = self.parent_method.na
        self.nvir = self.parent_method.nuocc

        self.fock_builder = self.system.fock_builder

    def _build_df_iaQ(self):
        """
        Build 3-index integrals (ia|Q) using density fitting.

        Returns:
        --------
            B_iaQ: 3-index integrals (ia|Q) as a numpy array of shape (n_occ, n_vir, n_aux).
        """
        C_occ = self.C[:, : self.nocc]  # Occupied MOs
        C_vir = self.C[:, self.nocc :]  # Virtual MOs

        B_Qia = self.fock_builder.B_tensor_gen_block(C_occ, C_vir)
        assert (
            B_Qia.shape[1] == self.nocc
        ), f"B occ mismatch: {B_Qia.shape[0]} vs {self.nocc}"
        assert (
            B_Qia.shape[2] == self.nvir
        ), f"B vir mismatch: {B_Qia.shape[1]} vs {self.nvir}"

        return B_Qia.transpose(1, 2, 0).copy()  # Shape (n_occ, n_vir, n_aux)

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
        t2 = np.empty((nocc, nocc, nvir, nvir))
        # antisym only in (a,b): t̃ = 2t - t^{ba}
        t2_as = np.empty_like(t2)

        E_corr = 0.0

        for i in range(nocc):
            Bi = B[i]  # (nvir, naux)
            for j in range(nocc):
                Bj = B[j]
                gijab = Bi @ Bj.T  # (a,b)
                denom = eps_i[i] + eps_i[j] - eps_a[:, None] - eps_a[None, :]
                tijab = self._safe_divide(gijab, denom)
                t2[i, j] = tijab
                t2_as[i, j] = 2.0 * tijab - tijab.T

                # energy contribution
                E_corr += np.sum((2.0 * gijab - gijab.T) * tijab)

        return t2, t2_as, E_corr


@dataclass
class ROHFMP2(MP2Base):
    """
    Density-Fitted Møller–Plesset perturbation theory (DF-MP2) method with ROHF canonical orbitals.

    Parameters
    ----------
    compute_1rdm
        If True, build the spin-free 1-RDM (unrelaxed MP2).
    compute_1rdm_ao
        If True, build the spin-free 1-RDM in AO basis.
    compute_2rdm
        If True, build the spin-free 2-RDM (potentially large).
    compute_cumulants
        If True, build 2-body cumulant (and 1-body hole RDM if needed).
    Returns
    -------
    float
        MP2 total energy (E_HF + E_corr).
    """

    def run(self):
        t0 = time.monotonic()

        self._startup()

        self.B_iaQ = self._build_df_iaQ()

        self.t2, self.t2_as, self.E_corr = self._build_t2_all(self.B_iaQ)

        self.E_total = self.parent_method.E + self.E_corr

        # ---- custom RDM pipeline ----
        self.gamma1_sf = None
        self.gamma2_sf = None
        self.lambda2_sf = None

        if self.compute_1rdm or self.compute_1rdm_ao:
            self.gamma1_a, self.gamma1_b = self.make_rohf_1rdm()

            self.gamma1_sf = self.gamma1_a + self.gamma1_b
            self.gamma1_sf = 0.5 * (self.gamma1_sf + self.gamma1_sf.T)

            if self.compute_1rdm_ao:
                self.gamma1_sf_ao = self.gamma1_mo_to_ao(self.gamma1_sf)

        if self.compute_2rdm or self.compute_cumulants:
            self.gamma2_sf = self.make_rohf_2rdm()

        if self.compute_cumulants:
            self.lambda2_sf = self.make_mp2_sf_2cumulants(
                self.gamma1_sf, self.gamma2_sf
            )

        logger.log_info1("ROHF-MP2 calculation completed.")
        logger.log_info1(f"E(corr) = {self.E_corr:.13f} Eh")
        logger.log_info1(f"E(total) = {self.E_total:.13f} Eh")
        logger.log_info1(f"||t2|| = {np.linalg.norm(self.t2)}")

        self.executed = True
        dt = time.monotonic() - t0
        logger.log_info1(f"Time = {dt:.3f} s")

        return self.E_total

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        # Copy system + MO information
        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        # MO reordering (contiguous correlated space)
        self.C = self.C[0].copy()
        self.eps = self.parent_method.eps[0].copy()

        self.docc = (self.parent_method.na + self.parent_method.nb) // 2
        self.socc = self.parent_method.na - self.docc
        self.nocc = self.docc + self.socc
        self.nvir = self.parent_method.nuocc

        self.fock_builder = self.system.fock_builder

    def _build_df_iaQ(self):
        nd = self.docc
        ns = self.socc

        C_occ = self.C[:, : nd + ns]
        C_vir = self.C[:, nd + ns :]

        B_Qia = self.fock_builder.B_tensor_gen_block(C_occ, C_vir)

        assert B_Qia.shape[1] == self.nocc
        assert B_Qia.shape[2] == self.nvir

        return B_Qia.transpose(1, 2, 0).copy()

    def _build_t2_all(self, B):
        nd = self.docc
        ns = self.socc
        nvir = self.nvir

        eps = self.eps
        eps_d = eps[:nd]
        eps_s = eps[nd : nd + ns]
        eps_v = eps[nd + ns :]

        E_corr = 0.0
        t2 = np.empty((self.nocc, self.nocc, nvir, nvir))
        t2_as = np.empty_like(t2)

        # doubly-doubly contribution
        for i in range(nd):
            Bi = B[i]  # (nvir, naux)
            for j in range(nd):
                Bj = B[j]
                gijab = Bi @ Bj.T  # (a,b)
                denom = eps_d[i] + eps_d[j] - eps_v[:, None] - eps_v[None, :]
                tijab = self._safe_divide(gijab, denom)

                t2[i, j] = tijab
                t2_as[i, j] = 2.0 * tijab - tijab.T

                E_corr += np.sum((2.0 * gijab - gijab.T) * tijab)

        # doubly-singly occupied contribution (only i or j in singly occupied block)
        for i in range(nd):
            Bi = B[i]  # (nvir, naux)
            for r in range(ns):
                idx_r = nd + r
                Br = B[idx_r]
                girab = Bi @ Br.T  # (a,b)
                denom = eps_s[r] + eps_d[i] - eps_v[:, None] - eps_v[None, :]
                tirab = self._safe_divide(girab, denom)

                t2[i, idx_r] = tirab
                t2[idx_r, i] = tirab.T

                t2_as[i, idx_r] = tirab
                t2_as[idx_r, i] = tirab.T

                E_corr += 2.0 * np.sum(girab * tirab)

        # singly-singly occupied contribution (i,j both in singly occupied block)
        for r in range(ns):
            idx_r = nd + r
            Br = B[idx_r]  # (nvir, naux)

            for s in range(ns):
                idx_s = nd + s
                Bs = B[idx_s]

                grsab = Br @ Bs.T  # (a,b)
                denom = eps_s[r] + eps_s[s] - eps_v[:, None] - eps_v[None, :]
                trsab = self._safe_divide(grsab, denom)

                t2[idx_r, idx_s] = trsab
                t2[idx_s, idx_r] = trsab.T

                t2_as[idx_r, idx_s] = trsab
                t2_as[idx_s, idx_r] = trsab.T

                factor = 0.5 if r == s else 1.0
                E_corr += factor * np.sum(grsab * trsab)

        return t2, t2_as, E_corr

    def make_rohf_1rdm(self):
        nd = self.docc
        ns = self.socc
        nocc = self.nocc
        nvir = self.nvir
        nmo = nocc + nvir

        gamma1_a = np.zeros((nmo, nmo))
        gamma1_b = np.zeros((nmo, nmo))

        # --- reference occupations ---
        # doubly occupied
        gamma1_a[:nd, :nd] += np.eye(nd)
        gamma1_b[:nd, :nd] += np.eye(nd)

        # singly occupied (alpha only)
        gamma1_a[nd : nd + ns, nd : nd + ns] += np.eye(ns)

        # --- MP2 corrections (reuse t2 structure) ---
        t2 = self.t2
        t2_as = self.t2_as

        t2_dd = t2[:nd, :nd]
        t2_as_dd = t2_as[:nd, :nd]

        doo = -0.5 * np.einsum("imef,jmef->ij", t2_as_dd, t2_dd)

        # vir-vir correction
        dvv = 0.5 * np.einsum("mnae,mnbe->ab", t2_as, t2, optimize=True)

        # alpha: doubly occupied block
        gamma1_a[:nd, :nd] += doo + doo.T

        gamma1_b[:nd, :nd] += doo[:nd, :nd] + doo[:nd, :nd].T  # beta only sees doubly

        gamma1_a[nocc:, nocc:] += dvv + dvv.T
        gamma1_b[nocc:, nocc:] += dvv + dvv.T

        return gamma1_a, gamma1_b

    def make_rohf_2rdm(self):
        nocc, nvir = self.nocc, self.nvir
        nmo = nocc + nvir

        dm2 = np.zeros((nmo, nmo, nmo, nmo))

        o = np.arange(nocc)
        v = np.arange(nocc, nmo)

        t2 = self.t2

        # OVOV block (same as RHF approx)
        dovov = (2.0 * t2.transpose(0, 2, 1, 3) - t2.transpose(0, 3, 1, 2)) * 2.0

        dm2[np.ix_(o, v, o, v)] = dovov
        dm2[np.ix_(v, o, v, o)] = dovov.transpose(1, 0, 3, 2)

        # add disconnected part from gamma1
        dm1 = self.gamma1_sf

        term1 = np.einsum("qp,sr->pqrs", dm1, dm1)
        term2 = np.einsum("qr,sp->pqrs", dm1, dm1)

        dm2 += term1 - 0.5 * term2

        return dm2


@dataclass
class UHFMP2(MP2Base):
    """
    Density-Fitted Møller–Plesset perturbation theory (DF-MP2) method with UHF canonical orbitals.

    Parameters
    ----------
    compute_1rdm
        If True, build the spin-free 1-RDM (unrelaxed MP2).
    compute_1rdm_ar
        If True, build the spin-free 1-RDM in AO basis.
    compute_2rdm
        If True, build the spin-free 2-RDM (potentially large).
    compute_cumulants
        If True, build 2-body cumulant (and 1-body hole RDM if needed).
        Usually implies compute_rdm2 unless you implement a direct cumulant builder.

    Returns
    -------
    float
        MP2 total energy (E_HF + E_corr).
    """

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

        self._startup()

        self.B_iaQ = self._build_df_iaQ()
        self.Ba_iaQ, self.Bb_iaQ = self.B_iaQ

        (self.t2_a, self.t2_b, self.t2_ab, self.E_corr) = self._build_t2_all(self.B_iaQ)

        self.E_total = self.parent_method.E + self.E_corr
        self.gamma1_sf = None
        self.gamma2_sf = None
        self.lambda2_sf = None
        # MP2Base assumes spin-restricted tensors. UHF overrides all RDM builders.
        if (
            self.compute_1rdm
            or self.compute_1rdm_ao
            or self.compute_2rdm
            or self.compute_cumulants
        ):
            if self.compute_1rdm or self.compute_1rdm_ao:
                self.make_mp2_sf_1rdm_intermediates(self.B_iaQ)

                if self.compute_1rdm_ao:
                    self.gamma1_sf_ao = self.gamma1_mo_to_ao(self.gamma1_sf)

        if self.compute_2rdm or self.compute_cumulants:
            self.gamma2_sf = self.make_mp2_sf_2rdm()

        if self.compute_cumulants:
            self.lambda2_sf = self.make_mp2_sf_2cumulants(
                self.gamma1_sf, self.gamma2_sf
            )

        logger.log_info1("UHF-MP2 calculation completed.")
        logger.log_info1(f"E(corr) = {self.E_corr:.13f} Eh")
        logger.log_info1(f"E(total) = {self.E_total:.13f} Eh")
        norm = (
            np.linalg.norm(self.t2_a)
            + np.linalg.norm(self.t2_b)
            + np.linalg.norm(self.t2_ab)
        )
        logger.log_info1(f"||t2|| = {norm}")

        self.executed = True
        dt = time.monotonic() - t0
        logger.log_info1(f"Time = {dt:.3f} s")
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

    def _build_t2_all(self, B):
        Ba, Bb = B

        eps_a_i = self.eps_a[: self.naocc]
        eps_a_a = self.eps_a[self.naocc :]
        eps_b_i = self.eps_b[: self.nbocc]
        eps_b_a = self.eps_b[self.nbocc :]

        naocc, navir = self.naocc, self.navir
        nbocc, nbvir = self.nbocc, self.nbvir

        # allocate
        t2_a = np.zeros((naocc, naocc, navir, navir))
        t2_b = np.zeros((nbocc, nbocc, nbvir, nbvir))
        t2_ab = np.zeros((naocc, nbocc, navir, nbvir))

        E_corr = 0.0
        tiny = 1e-12

        # =========================
        # ALPHA-ALPHA
        # =========================
        for i in range(naocc):
            Bi = Ba[i]  # (navir, naux)
            for j in range(naocc):
                Bj = Ba[j]

                # (ia|Q)(jb|Q) → (ab)
                gijab = Bi @ Bj.T
                gijba = gijab.T

                # antisymmetrized integrals
                g_as = gijab - gijba

                denom = eps_a_i[i] + eps_a_i[j] - eps_a_a[:, None] - eps_a_a[None, :]

                tijab = self._safe_divide(g_as, denom, tiny)

                t2_a[i, j] = tijab

                # energy (same-spin → 1/4 factor)
                E_corr += 0.25 * np.sum(g_as * tijab)

        # =========================
        # BETA-BETA
        # =========================
        for i in range(nbocc):
            Bi = Bb[i]
            for j in range(nbocc):
                Bj = Bb[j]

                gijab = Bi @ Bj.T
                gijba = gijab.T

                g_as = gijab - gijba

                denom = eps_b_i[i] + eps_b_i[j] - eps_b_a[:, None] - eps_b_a[None, :]

                tijab = self._safe_divide(g_as, denom, tiny)

                t2_b[i, j] = tijab

                E_corr += 0.25 * np.sum(g_as * tijab)

        # =========================
        # ALPHA-BETA
        # =========================
        Ea = eps_a_a[:, None]
        Eb = eps_b_a[None, :]

        for i in range(naocc):
            Bi = Ba[i]
            for j in range(nbocc):
                Bj = Bb[j]

                gijab = Bi @ Bj.T  # no antisymmetrization

                denom = eps_a_i[i] + eps_b_i[j] - Ea - Eb

                tijab = self._safe_divide(gijab, denom, tiny)

                t2_ab[i, j] = tijab

                # opposite-spin → no 1/4 factor
                E_corr += np.sum(gijab * tijab)

        return t2_a, t2_b, t2_ab, E_corr

    def make_mp2_sf_1rdm_intermediates(self, B):
        # Ensure amplitudes exist
        if not all(hasattr(self, attr) for attr in ("t2_a", "t2_b", "t2_ab")):
            self.t2_a, self.t2_b, self.t2_ab, _ = self._build_t2_all(B)

        t2_a = self.t2_a
        t2_b = self.t2_b
        t2_ab = self.t2_ab

        naocc, navir = self.naocc, self.navir
        nbocc, nbvir = self.nbocc, self.nbvir

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

    def gamma1_mo_to_ao(self, gamma1_sf):
        return self.C @ gamma1_sf @ self.C.T

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
        dm2_aa = np.zeros((nmo, nmo, nmo, nmo))
        dm2_bb = np.zeros((nmo, nmo, nmo, nmo))
        dm2_ab = np.zeros((nmo, nmo, nmo, nmo))

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


class MP2MCASolverLike:
    def __init__(
        self,
        gamma1_sf: np.ndarray,
        lambda2_sf: np.ndarray,
        U: np.ndarray | None = None,
        orbital_indices=None,
    ):
        norb = gamma1_sf.shape[0]

        if U is not None:
            # Rotate 1-RDM and 2-cumulant ONCE into NO basis
            Γ1 = U.T @ gamma1_sf @ U
            λsf_no = np.einsum(
                "pqrs,pi,qj,rk,sl->ijkl", lambda2_sf, U, U, U, U, optimize=True
            )

            # Map spin-free cumulant into λab only (your convention)
            self.λaa = np.zeros((norb, norb, norb, norb), dtype=lambda2_sf.dtype)
            self.λbb = np.zeros_like(self.λaa)
            self.λab = 0.5 * λsf_no

            self.Γ1 = Γ1
            self.orbital_indices = list(range(norb))  # NO labels
            self.U_no = U

        else:
            self.Γ1 = gamma1_sf
            self.λaa = np.zeros((norb, norb, norb, norb), dtype=lambda2_sf.dtype)
            self.λbb = np.zeros_like(self.λaa)
            self.λab = 0.5 * lambda2_sf

            self.orbital_indices = (
                list(orbital_indices)
                if orbital_indices is not None
                else list(range(norb))
            )
