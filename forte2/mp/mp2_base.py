from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import sys
import time
import resource

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
    store_t2: bool = False
    executed: bool = field(default=False, init=False)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        if not isinstance(parent_method, (RHF, ROHF, UHF)):
            raise TypeError("MP2 requires an RHF, ROHF, or UHF reference.")
        return self

    def run(self):
        t0 = time.monotonic()
        mem0 = self._memory_snapshot()

        self._startup()
        need_t2 = self._needs_t2_storage()

        self.B_iaQ = self._build_df_iaQ()

        self.t2, self.t2_as, self.E_corr = self._build_t2_all(
            self.B_iaQ, store_t2=need_t2
        )

        self.E_total = self.parent_method.E + self.E_corr

        self._postprocess_rdms()

        self.executed = True
        self._log_completion(time.monotonic() - t0, self._t2_norm(), mem0)
        return self.E_total

    def _initialize_rdm_outputs(self):
        self.gamma1_sf = None
        self.gamma1_sf_ao = None
        self.gamma2_sf = None
        self.lambda2_sf = None

    def _needs_t2_storage(self) -> bool:
        return (
            self.store_t2
            or self.compute_1rdm
            or self.compute_1rdm_ao
            or self.compute_2rdm
            or self.compute_cumulants
        )

    def _postprocess_rdms(self):
        self._initialize_rdm_outputs()
        need_gamma1 = (
            self.compute_1rdm
            or self.compute_1rdm_ao
            or self.compute_2rdm
            or self.compute_cumulants
        )

        if need_gamma1:
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

    def _memory_snapshot(self):
        return {
            "peak_rss_mb": self._peak_rss_mb(),
            "current_rss_mb": self._current_rss_mb(),
        }

    def _peak_rss_mb(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return peak / 1024**2
        return peak / 1024.0

    def _current_rss_mb(self):
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2

    def _log_memory(self, start_mem):
        end_peak = self._peak_rss_mb()
        peak_delta = max(0.0, end_peak - start_mem["peak_rss_mb"])
        logger.log_info1(f"Peak RSS = {end_peak:.1f} MB (delta {peak_delta:.1f} MB)")

        end_current = self._current_rss_mb()
        start_current = start_mem["current_rss_mb"]
        if end_current is not None:
            if start_current is not None:
                current_delta = end_current - start_current
                logger.log_info1(
                    f"Current RSS = {end_current:.1f} MB (delta {current_delta:.1f} MB)"
                )
            else:
                logger.log_info1(f"Current RSS = {end_current:.1f} MB")

    def _log_completion(self, elapsed: float, t2_norm: float | None, start_mem):
        logger.log_info1(f"{self._reference_label()}-MP2 calculation completed.")
        logger.log_info1(f"E(corr) = {self.E_corr:.13f} Eh")
        logger.log_info1(f"E(total) = {self.E_total:.13f} Eh")
        if t2_norm is None:
            logger.log_info1("||t2|| = not stored")
        else:
            logger.log_info1(f"||t2|| = {t2_norm}")
        self._log_memory(start_mem)
        logger.log_info1(f"Time = {elapsed:.3f} s")

    def _reference_label(self) -> str:
        return "MP2"

    def _t2_norm(self):
        if getattr(self, "t2", None) is None:
            return None
        return np.linalg.norm(self.t2)

    def _copy_restricted_reference(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)

        self.C = self.C[0].copy()
        self.eps = self.parent_method.eps[0].copy()
        self.fock_builder = self.system.fock_builder

    def _build_restricted_df_iaQ(self, nocc: int):
        C_occ = self.C[:, :nocc]
        C_vir = self.C[:, nocc:]

        B_Qia = self.fock_builder.B_tensor_gen_block(C_occ, C_vir)
        assert B_Qia.shape[1] == nocc, f"B occ mismatch: {B_Qia.shape[1]} vs {nocc}"
        assert (
            B_Qia.shape[2] == self.nvir
        ), f"B vir mismatch: {B_Qia.shape[2]} vs {self.nvir}"

        return B_Qia.transpose(1, 2, 0).copy()

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

    def _safe_divide(self, num, denom, out=None, tiny=1e-12, label="MP2 denom"):
        mask = np.abs(denom) < tiny
        n_bad = np.count_nonzero(mask)
        if n_bad:
            logger.log_warning(f"{label}: {n_bad} / {denom.size} elements < {tiny:g}")
        if out is None:
            return np.divide(num, np.where(mask, np.inf, denom))
        else:
            return np.divide(num, np.where(mask, np.inf, denom), out=out)

    @abstractmethod
    def _build_df_iaQ(self): ...

    @abstractmethod
    def _startup(self): ...

    @abstractmethod
    def _build_t2_all(self, B, store_t2=True): ...

