from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

import numpy as np

from forte2.base_classes import SystemMixin, MOsMixin
from forte2.scf import RHF, ROHF, UHF
from forte2.helpers import logger


@dataclass
class DFMP2Base(SystemMixin, MOsMixin, ABC):
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

    Raises
    ------
    TypeError
        If the provided reference is not a supported SCF method
        (e.g., RHF or ROHF).
    RuntimeError
        If required reference data (orbitals, energies, integrals)
        are unavailable or inconsistent.
    NotImplementedError
        If a subclass does not implement required amplitude-building
        routines.
    """

    compute_1rdm: bool = False
    compute_1rdm_ao: bool = False
    compute_2rdm: bool = False
    compute_cumulants: bool = False
    executed: bool = field(default=False, init=False)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        if not isinstance(parent_method, (RHF, ROHF)):
            raise TypeError("DFMP2 requires an RHF or ROHF reference.")
        return self

    def run(self):
        self._startup()

        self.B_iaQ = self._build_df_iaQ()

        self.t2, self.t2_as, self.E_corr = self._build_t2_all(self.B_iaQ)

        self.E_total = self.parent_method.E + self.E_corr

        self._postprocess_rdms()

        self.executed = True
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

        return B_Qia.transpose(1, 2, 0).copy()  # Shape (n_occ, n_vir, n_aux)

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
            tiny = 1e-12
            mask = np.abs(denom) < tiny
            n_bad = np.count_nonzero(mask)
            if n_bad:
                logger.log_warning(
                    f"MP2 denom clamp: {n_bad} / {denom.size} elements < {tiny:g}"
                )
            denom = np.where(mask, np.inf, denom)
            t2i = g_jab / denom  # (j,a,b)

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

    @abstractmethod
    def _startup(self): ...

    @abstractmethod
    def _build_t2_all(self, B): ...


@dataclass
class DFRHFMP2(DFMP2Base):
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
                tiny = 1e-12
                mask = np.abs(denom) < tiny
                n_bad = np.count_nonzero(mask)
                if n_bad:
                    logger.log_warning(
                        f"MP2 denom clamp: {n_bad} / {denom.size} elements < {tiny:g}"
                    )
                denom = np.where(mask, np.inf, denom)
                tijab = gijab / denom

                t2[i, j] = tijab
                t2_as[i, j] = 2.0 * tijab - tijab.T

                # energy contribution
                E_corr += np.sum((2.0 * gijab - gijab.T) * tijab)

        return t2, t2_as, E_corr


@dataclass
class DFROHFMP2(DFMP2Base):
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

    def _build_t2_all(self, B):
        nd = self.ndocc
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
                tiny = 1e-12
                mask = np.abs(denom) < tiny
                n_bad = np.count_nonzero(mask)
                if n_bad:
                    logger.log_warning(
                        f"MP2 denom clamp: {n_bad} / {denom.size} elements < {tiny:g}"
                    )
                denom = np.where(mask, np.inf, denom)
                tijab = gijab / denom

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
                tiny = 1e-12
                mask = np.abs(denom) < tiny
                n_bad = np.count_nonzero(mask)
                if n_bad:
                    logger.log_warning(
                        f"MP2 denom clamp: {n_bad} / {denom.size} elements < {tiny:g}"
                    )
                denom = np.where(mask, np.inf, denom)
                tirab = girab / denom

                t2[i, idx_r] = tirab
                t2_as[i, idx_r] = tirab  # no antisymmetry for single occupancy

                E_corr += np.sum(girab * tirab)

        # singly-singly occupied contribution (i,j both in singly occupied block)
        for r in range(ns):
            idx_r = nd + r
            Br = B[idx_r]  # (nvir, naux)

            for s in range(ns):
                idx_s = nd + s
                Bs = B[idx_s]

                grsab = Br @ Bs.T  # (a,b)
                denom = eps_s[r] + eps_s[s] - eps_v[:, None] - eps_v[None, :]
                tiny = 1e-12
                mask = np.abs(denom) < tiny
                n_bad = np.count_nonzero(mask)
                if n_bad:
                    logger.log_warning(
                        f"MP2 denom clamp: {n_bad} / {denom.size} elements < {tiny:g}"
                    )
                denom = np.where(mask, np.inf, denom)
                trsab = grsab / denom

                t2[idx_r, idx_s] = trsab
                t2_as[idx_r, idx_s] = trsab  # no antisymmetry for single occupancy

                factor = 0.5 if r == s else 1.0
                E_corr += factor * np.sum(grsab * trsab)
        return t2, t2_as, E_corr


@dataclass
class DFUHFMP2(DFMP2Base):
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

        self.naocc = self.parent_method.na
        self.navir = self.parent_method.nuocc

        self.nbocc = self.parent_method.nb
        self.nbvir = self.parent_method.nvocc

        self.fock_builder = self.system.fock_builder

    def _build_df_iaQ(self, Ba, Bb):
        raise NotImplementedError("DF-UHF-MP2 df_iaQ builder not implemented yet.")

    def _build_t2_all(self, Ba, Bb):
        raise NotImplementedError("DF-UHF-MP2 t2 builder not implemented yet.")


# @dataclass
# class DFUHFMP2(SystemMixin, MOsMixin, ABC):
#     """
#     Density-Fitted Møller–Plesset perturbation theory (DF-MP2) method with UHF canonical orbitals.

#     Parameters
#     ----------
#     compute_1rdm
#         If True, build the spin-free 1-RDM (unrelaxed MP2).
#     compute_1rdm_ar
#         If True, build the spin-free 1-RDM in AO basis.
#     compute_2rdm
#         If True, build the spin-free 2-RDM (potentially large).
#     compute_cumulants
#         If True, build 2-body cumulant (and 1-body hole RDM if needed).
#         Usually implies compute_rdm2 unless you implement a direct cumulant builder.

#     Returns
#     -------
#     float
#         MP2 total energy (E_HF + E_corr).
#     """

#     compute_1rdm: bool = False
#     compute_1rdm_ao: bool = False
#     compute_2rdm: bool = False
#     compute_cumulants: bool = False
#     executed: bool = False

#     def __call__(self, parent_method):
#         self.parent_method = parent_method
#         assert isinstance(
#             self.parent_method, UHF
#         ), "Parent method must be of UHF reference."
#         return self

#     def _startup(self):
#         if not self.parent_method.executed:
#             self.parent_method.run()

#     # Copy system + MO information
#     SystemMixin.copy_from_upstream(self, self.parent_method)
#     MOsMixin.copy_from_upstream(self, self.parent_method)

#     # MO reordering (contiguous correlated space)
#     self.Ca = self.C[0].copy()
#     self.Cb = self.C[1].copy()
#     self.eps_a = self.parent_method.eps[0].copy()
#     self.eps_b = self.parent_method.eps[1].copy()

#     self.naocc = self.parent_method.na
#     self.navir = self.parent_method.nuocc

#     self.nbocc = self.parent_method.nb
#     self.nbvir = self.parent_method.nvocc

#     self.fock_builder = self.system.fock_builder

# def run(self):
#     """
#     Run DF-UHF-MP2.
#     """
#     import time

#     t0 = time.monotonic()
#     logger.log_info1("Starting DF-UHF-MP2 calculation.")

#     self._startup()

#     self.Ba_iaQ, self.Bb_iaQ = self._build_df_iaQ()  # shape (nocc, nvir, naux)

#     # --- energies
#     (
#         self.t2,
#         self.t2_as,
#         self.t2_b,
#         self.t2_b_as,
#         self.t2_ab,
#         self.t2_ab_as,
#         self.E_corr,
#     ) = self._build_t2_all(Ba=self.Ba_iaQ, Bb=self.Bb_iaQ)
#     self.E_total = self.parent_method.E + self.E_corr

#     # --- optional density info
#     self.gamma1_sf = None
#     self.gamma2_sf = None
#     self.lambda2_sf = None

#     logger.log_info1("DF-RHF-MP2 calculation completed.")
#     logger.log_info1(f"E(corr)  = {self.E_corr:.13f} Eh")
#     logger.log_info1(f"E(total) = {self.E_total:.13f} Eh")
#     logger.log_info1(f"||t2|| = {np.linalg.norm(self.t2)}")

#     if self.compute_1rdm:
#         self.gamma1_a_sf, self.gamma1_b_sf = self.make_mp2_sf_1rdm_intermediates(
#             Ba=self.Ba_iaQ, Bb=self.Bb_iaQ
#         )
#         if self.compute_1rdm_ao:
#             self.gamma1_sf_ao = self.gamma1_mo_to_ao(
#                 self.gamma1_a_sf + self.gamma1_b_sf
#             )

#     if self.compute_2rdm or self.compute_cumulants:
#         self.gamma2_sf = self.make_mp2_sf_2rdm(
#             t2_a=self.t2_as,
#             t2_b=self.t2_b_as,
#             dm1a=self.gamma1_a_sf,
#             dm1b=self.gamma1_b_sf,
#         )

#     self.executed = True
#     dt = time.monotonic() - t0

#     logger.log_info1(f"Time     = {dt:.3f} s")

#     return self.E_total

# def _build_t2_all(self, Ba, Bb):
#     """
#     Build all MP2 amplitudes t_{ij}^{ab} and antisymmetrized t̃_{ij}^{ab}.

#     Shapes
#     ------
#     B: (nocc, nvir, naux)
#     t2, t2_as: (nocc, nocc, nvir, nvir)

#     Energy
#     ------
#     E_corr = Σ_{ijab} (2(ia|jb) - (ib|ja)) * (ia|jb) / Δ_{ij}^{ab}
#         = Σ_{ijab} (2 g_{ij}^{ab} - g_{ij}^{ba}) * t_{ij}^{ab}
#     """
#     eps_a_i = self.eps_a[: self.naocc]
#     eps_a_a = self.eps_a[self.naocc :]

#     eps_b_i = self.eps_b[: self.nbocc]
#     eps_b_a = self.eps_b[self.nbocc :]
#     naocc, navir = self.naocc, self.navir
#     nbocc, nbvir = self.nbocc, self.nbvir
#     t2_a = np.empty((naocc, naocc, navir, navir))
#     t2_b = np.empty((nbocc, nbocc, nbvir, nbvir))
#     # antisym only in (a,b): t̃ = 2t - t^{ba}
#     t2_a_as = np.empty_like(t2_a)
#     t2_b_as = np.empty_like(t2_b)

#     E_corr = 0.0

#     for i in range(naocc):
#         Bi = Ba[i]  # (navir, naux)
#         for j in range(naocc):
#             Bj = Ba[j]
#             gijab_a = Bi @ Bj.T  # (navir, navir)
#             denom_a = eps_a_i[i] + eps_a_i[j] - eps_a_a[:, None] - eps_a_a[None, :]
#             tiny = 1e-12
#             mask = np.abs(denom_a) < tiny
#             n_bad = np.count_nonzero(mask)
#             if n_bad:
#                 logger.log_warning(
#                     f"MP2 denom clamp: {n_bad} / {denom_a.size} elements < {tiny:g}"
#                 )
#             denom_a = np.where(mask, np.inf, denom_a)
#             tijab_a = gijab_a / denom_a

#             t2_a[i, j] = tijab_a
#             t2_a_as[i, j] = tijab_a - tijab_a.T

#             # energy contribution
#             E_corr += np.sum((gijab_a - gijab_a.T) * tijab_a)

#     for i in range(nbocc):
#         Bi = Bb[i]  # (nbvir, naux)
#         for j in range(nbocc):
#             Bj = Bb[j]
#             gijab_b = Bi @ Bj.T  # (nbvir, nbvir)
#             denom_b = eps_b_i[i] + eps_b_i[j] - eps_b_a[:, None] - eps_b_a[None, :]
#             tiny = 1e-12
#             mask = np.abs(denom_b) < tiny
#             n_bad = np.count_nonzero(mask)
#             if n_bad:
#                 logger.log_warning(
#                     f"MP2 denom clamp: {n_bad} / {denom_b.size} elements < {tiny:g}"
#                 )
#             denom_b = np.where(mask, np.inf, denom_b)
#             tijab_b = gijab_b / denom_b

#             t2_b[i, j] = tijab_b
#             t2_b_as[i, j] = tijab_b - tijab_b.T
#             E_corr += np.sum((gijab_b - gijab_b.T) * tijab_b)

#     # Build alpha-beta integrals
#     Ea = eps_a_a[:, None]
#     Eb = eps_b_a[None, :]
#     for i in range(naocc):
#         Bi = Ba[i]  # (navir, naux)
#         for j in range(nbocc):
#             Bj = Bb[j]  # (nbvir, naux)

#             gijab_ab = Bi @ Bj.T  # (navir, nbvir)

#             denom_ab = eps_a_i[i] + eps_b_i[j] - Ea - Eb  # (a,b)

#             tiny = 1e-12
#             mask = np.abs(denom_ab) < tiny
#             n_bad = np.count_nonzero(mask)
#             if n_bad:
#                 logger.log_warning(
#                     f"MP2 denom clamp: {n_bad} / {denom_ab.size} elements < {tiny:g}"
#                 )

#             denom_ab = np.where(mask, np.inf, denom_ab)

#             t2_ab = gijab_ab / denom_ab
#             E_corr += np.sum(gijab_ab * t2_ab)

#     return t2_a, t2_a_as, t2_b, t2_b_as, t2_ab, E_corr

# def _build_df_iaQ(self):
#     """
#     Build 3-index integrals (ia|Q) using density fitting.

#     Returns:
#     --------
#         B_iaQ alpha: 3-index integrals (ia|Q) as a numpy array of shape (n_occ, n_vir, n_aux).
#         B_iaQ beta: 3-index integrals (ia|Q) as a numpy array of shape (n_occ, n_vir, n_aux).
#     """
#     Ca_occ = self.Ca[:, : self.naocc]  # Alpha Occupied MOs
#     Ca_vir = self.Ca[:, self.naocc :]  # Alpha Virtual MOs

#     Cb_occ = self.Cb[:, : self.nbocc]  # Beta Occupied MOs
#     Cb_vir = self.Cb[:, self.nbocc :]  # Beta Virtual MOs

#     Ba_Qia = self.fock_builder.B_tensor_gen_block(Ca_occ, Ca_vir)
#     Bb_Qia = self.fock_builder.B_tensor_gen_block(Cb_occ, Cb_vir)

#     return (
#         Ba_Qia.transpose(1, 2, 0).copy(),
#         Bb_Qia.transpose(1, 2, 0).copy(),
#     )

# def make_mp2_sf_1rdm_intermediates(
#     self, Ba: np.ndarray, Bb: np.ndarray
# ) -> np.ndarray:
#     """
#     Fast unrelaxed spin-summed MP2 1-RDM (PySCF-style) using DF factors B[i,a,Q].
#     No full t2 tensor is formed.

#     Returns gamma with:
#     gamma_oo = 2I - (dm1occ + dm1occ.T)
#     gamma_vv =      (dm1vir + dm1vir.T)
#     where dm1occ, dm1vir match PySCF's _gamma1_intermediates (up to symmetry handling).
#     """
#     eps_a_o = self.eps_a[: self.naocc]
#     eps_b_o = self.eps_b[: self.nbocc]
#     eps_a_v = self.eps_a[self.naocc :]
#     eps_b_v = self.eps_b[self.nbocc :]
#     naocc, navir = self.naocc, self.navir
#     nbocc, nbvir = self.nbocc, self.nbvir
#     namo = naocc + navir
#     nbmo = nbocc + nbvir

#     dm1aocc = np.zeros((naocc, naocc))
#     dm1avir = np.zeros((navir, navir))

#     dm1bocc = np.zeros((nbocc, nbocc))
#     dm1bvir = np.zeros((nbvir, nbvir))

#     ea_ab = eps_a_v[:, None] + eps_a_v[None, :]  # (a,b)
#     eb_ab = eps_b_v[:, None] + eps_b_v[None, :]  # (b,a)

#     for i in range(naocc):
#         # g_jab = (ia|jb) for all j,a,b via DF
#         # B[i]: (a,Q), B: (j,b,Q)
#         g_jab = np.einsum("aQ,jbQ->jab", Ba[i], Ba, optimize=True)  # (j,a,b)
#         denom = (eps_a_o[i] + eps_a_o[:, None, None]) - ea_ab[None, :, :]  # (j,a,b)
#         tiny = 1e-12
#         mask = np.abs(denom) < tiny
#         n_bad = np.count_nonzero(mask)
#         if n_bad:
#             logger.log_warning(
#                 f"MP2 denom clamp: {n_bad} / {denom.size} elements < {tiny:g}"
#             )
#         denom = np.where(mask, np.inf, denom)
#         t2i = g_jab / denom  # (j,a,b)

#         l2i = t2i  # real case; for complex use t2i.conj()

#         # dm1vir += einsum('jca,jcb->ba', l2i, t2i) - einsum('jca,jbc->ba', l2i, t2i)
#         dm1avir += np.einsum("jca,jcb->ba", l2i, t2i, optimize=True) - np.einsum(
#             "jca,jbc->ba", l2i, t2i, optimize=True
#         )

#         # dm1occ += einsum('iab,jab->ij', l2i, t2i) - einsum('iab,jba->ij', l2i, t2i)
#         dm1aocc += np.einsum("iab,jab->ij", l2i, t2i, optimize=True) - np.einsum(
#             "iab,jba->ij", l2i, t2i, optimize=True
#         )
#     for i in range(nbocc):
#         # g_jab = (ia|jb) for all j,a,b via DF
#         # B[i]: (a,Q), B: (j,b,Q)
#         g_jab = np.einsum("aQ,jbQ->jab", Bb[i], Bb, optimize=True)  # (j,a,b)
#         denom = (eps_b_o[i] + eps_b_o[:, None, None]) - eb_ab[None, :, :]  # (j,a,b)
#         tiny = 1e-12
#         mask = np.abs(denom) < tiny
#         n_bad = np.count_nonzero(mask)
#         if n_bad:
#             logger.log_warning(
#                 f"MP2 denom clamp: {n_bad} / {denom.size} elements < {tiny:g}"
#             )
#         denom = np.where(mask, np.inf, denom)
#         t2i = g_jab / denom  # (j,a,b)

#         l2i = t2i  # real case; for complex use t2i.conj()

#         # dm1vir += einsum('jca,jcb->ba', l2i, t2i) - einsum('jca,jbc->ba', l2i, t2i)
#         dm1bvir += np.einsum("jca,jcb->ba", l2i, t2i, optimize=True) - np.einsum(
#             "jca,jbc->ba", l2i, t2i, optimize=True
#         )

#         # dm1occ += einsum('iab,jab->ij') - einsum('iab,jba->ij')
#         dm1bocc += np.einsum("iab,jab->ij", l2i, t2i, optimize=True) - np.einsum(
#             "iab,jba->ij", l2i, t2i, optimize=True
#         )

#     for i in range(naocc):
#         Bi = Ba[i]  # (navir, naux)
#         for j in range(nbocc):
#             Bj = Bb[j]  # (nbvir, naux)

#             gijab_ab = Bi @ Bj.T

#             denom_ab = eps_a_o[i] + eps_b_o[j] - eps_a_v[:, None] - eps_b_v[None, :]

#             tiny = 1e-12
#             mask = np.abs(denom_ab) < tiny
#             n_bad = np.count_nonzero(mask)
#             if n_bad:
#                 logger.log_warning(
#                     f"MP2 denom clamp: {n_bad} / {denom_ab.size} elements < {tiny:g}"
#                 )

#             denom_ab = np.where(mask, np.inf, denom_ab)

#             t2_ab = gijab_ab / denom_ab

#             dm1aocc += np.einsum("iab,jab->ij", t2_ab, t2_ab, optimize=True)
#             dm1bocc += np.einsum("iab,jab->ij", t2_ab, t2_ab, optimize=True)
#             dm1abocc += np.einsum("iab,jab->ij", t2_ab, t2_ab, optimize=True)
#             dm1avir += np.einsum("iab,jab->ab", t2_ab, t2_ab, optimize=True)
#             dm1bvir += np.einsum("iab,jab->ab", t2_ab, t2_ab, optimize=True)

#     gamma_a = np.zeros((namo, namo))
#     gamma_b = np.zeros((nbmo, nbmo))
#     gamma_a[:naocc, :naocc] = np.eye(naocc) - (dm1aocc + dm1aocc.T)
#     gamma_a[naocc:, naocc:] = dm1avir + dm1avir.T

#     gamma_b[:nbocc, :nbocc] = np.eye(nbocc) - (dm1bocc + dm1bocc.T)
#     gamma_b[nbocc:, nbocc:] = dm1bvir + dm1bvir.T
#     return gamma_a, gamma_b

# def gamma1_mo_to_ao(
#     self, gamma1_a_mo: np.ndarray, gamma1_b_mo: np.ndarray
# ) -> np.ndarray:
#     """
#     Transform spin-free 1-RDM from MO basis to AO basis.

#     Convention:
#     MO coefficients C are AO->MO: phi_p = sum_mu C[mu,p] chi_mu
#     gamma_AO = C gamma_MO C^T  (real orbitals; use C.conj() for complex)
#     """
#     Ca = self.Ca  # shape (nao, nmo)
#     Cb = self.Cb  # shape (nao, nmo)
#     return Ca @ gamma1_a_mo @ Ca.T + Cb @ gamma1_b_mo @ Cb.T

# def make_mp2_sf_2rdm(self, t2_a, t2_b, t2_ab, dm1a, dm1b):
#     naocc, navir = self.naocc, self.navir
#     nbocc, nbvir = self.nbocc, self.nbvir
#     namo = naocc + navir
#     nbmo = nbocc + nbvir

#     dm2_a = np.zeros((namo, namo, namo, namo), dtype=t2_a.dtype)
#     dm2_b = np.zeros((nbmo, nbmo, nbmo, nbmo), dtype=t2_b.dtype)
#     dm2_ab = np.zeros((namo, nbmo, namo, nbmo), dtype=t2_ab.dtype)
#     o_a = np.arange(naocc)
#     v_a = np.arange(naocc, namo)
#     o_b = np.arange(nbocc)
#     v_b = np.arange(nbocc, nbmo)

#     # -------------------------
#     # (1) OVOV / VOVO from t2
#     # -------------------------
#     # Build dovov_iajb with shape (i,a,j,b)
#     # dovov = (t2[i,j,a,b] - t2[i,j,b,a]), but rearranged to (i,a,j,b)
#     dovov_a_iajb = t2_a.transpose(0, 2, 1, 3) - t2_a.transpose(0, 3, 1, 2)
#     dovov_b_iajb = t2_b.transpose(0, 2, 1, 3) - t2_b.transpose(0, 3, 1, 2)
#     dovov_ab_iajb = t2_ab.transpose(0, 2, 1, 3)
#     # Fill dm2[o, v, o, v]
#     dm2_a[np.ix_(o_a, v_a, o_a, v_a)] = dovov_a_iajb
#     dm2_b[np.ix_(o_b, v_b, o_b, v_b)] = dovov_b_iajb  # same shape for beta block
#     dm2_ab[np.ix_(o_a, v_b, o_a, v_b)] = dovov_ab_iajb

#     # For real case: dm2[v,i,v,j] = dovov[a,j,b] with axes (a,b,j)
#     dm2_a[np.ix_(v_a, o_a, v_a, o_a)] = dovov_a_iajb.transpose(1, 0, 3, 2)
#     dm2_b[np.ix_(v_b, o_b, v_b, o_b)] = dovov_b_iajb.transpose(1, 0, 3, 2)
#     dm2_ab[np.ix_(v_a, o_b, v_a, o_b)] = dovov_ab_iajb.transpose(1, 0, 3, 2)
#     # -------------------------
#     # (2) Addback from dm1 + constants
#     # -------------------------
#     dm1a_work = dm1a.copy()
#     dm1a_work[o_a, o_a] -= 1.0  # subtract 2 on occupied diagonal only
#     dm1aT = dm1a_work.T

#     dm1b_work = dm1b.copy()
#     dm1b_work[o_b, o_b] -= 1.0  # subtract 2 on occupied diagonal only
#     dm1bT = dm1b_work.T

#     o_a = np.arange(naocc)
#     o_b = np.arange(nbocc)

#     dm2_a[o_a[:, None], o_a[:, None], :, :] += dm1aT[
#         None, :, :
#     ]  # (nocc,nmo,nmo) += (1,nmo,nmo)
#     dm2_a[:, :, o_a[None, :], o_a[None, :]] += dm1aT[
#         :, :, None
#     ]  # (nmo,nmo,nocc) += (nmo,nmo,1)
#     dm2_a[:, o_a[:, None], o_a[None, :], :] -= dm1aT[
#         :, None, :
#     ]  # (nmo,nocc,nmo) -= (nmo,1,nmo)
#     dm2_a[o_a[None, :], :, :, o_a[:, None]] -= dm1a_work[
#         None, :, :
#     ]  # (nocc,nmo,nmo) -= (1,nmo,nmo)

#     dm2_b[o_b[:, None], o_b[:, None], :, :] += dm1bT[
#         None, :, :
#     ]  # (nocc,nmo,nmo) += (1,nmo,nmo)
#     dm2_b[:, :, o_b[None, :], o_b[None, :]] += dm1bT[
#         :, :, None
#     ]  # (nmo,nmo,nocc) += (nmo,nmo,1)
#     dm2_b[:, o_b[:, None], o_b[None, :], :] -= dm1bT[
#         :, None, :
#     ]  # (nmo,nocc,nmo) -= (nmo,1,nmo)
#     dm2_b[o_b[None, :], :, :, o_b[:, None]] -= dm1b_work[
#         None, :, :
#     ]  # (nocc,nmo,nmo) -= (1,nmo,nmo)

#     # constants:
#     # dm2[i,i,j,j] += 1
#     dm2_a[o_a[:, None], o_a[:, None], o_a[None, :], o_a[None, :]] += 1.0
#     dm2_b[o_b[:, None], o_b[:, None], o_b[None, :], o_b[None, :]] += 1.0
#     # dm2[i,j,j,i] -= 1
#     dm2_a[o_a[:, None], o_a[None, :], o_a[None, :], o_a[:, None]] -= 1.0
#     dm2_b[o_b[:, None], o_b[None, :], o_b[None, :], o_b[:, None]] -= 1.0

#     return dm2_a, dm2_b, dm2_ab

# def make_mp2_sf_2cumulants(
#     self, gamma1_a, gamma1_b, gamma2_a, gamma2_b, gamma_2_ab
# ):
#     """
#     PySCF conventions:
#     dm1[q,p] = <p† q>
#     dm2[p,q,r,s] = < p† r† s q >

#     Disconnected/HF-like part:
#     dm2^(0)[p,q,r,s] = dm1[q,p]*dm1[s,r] - 1/2 * dm1[q,r]*dm1[s,p]
#     """
#     dm1_a = gamma1_a
#     dm1_b = gamma1_b
#     dm2_a = gamma2_a
#     dm2_b = gamma2_b
#     dm2_ab = gamma_2_ab

#     term1_a = np.einsum("qp,sr->pqrs", dm1_a, dm1_a, optimize=True)
#     term2_a = np.einsum("qr,sp->pqrs", dm1_a, dm1_a, optimize=True)
#     dm2_0_a = term1_a - 0.5 * term2_a

#     term1_b = np.einsum("qp,sr->pqrs", dm1_b, dm1_b, optimize=True)
#     term2_b = np.einsum("qr,sp->pqrs", dm1_b, dm1_b, optimize=True)
#     dm2_0_b = term1_b - 0.5 * term2_b

#     dm2_0_ab = np.einsum("qp,sr->pqrs", dm1_a, dm1_b, optimize=True)

#     lambda_aa = dm2_a - dm2_0_a
#     lambda_bb = dm2_b - dm2_0_b
#     lambda_ab = dm2_ab - dm2_0_ab
#     lambda_total = lambda_aa + lambda_bb + lambda_ab

#     return lambda_total

# def mp2_E_given_rdms(self, Ecore, H, V, gamma1_a, gamma1_b, gamma2_a, gamma2_b):
#     """
#     Computes mp2 Energy from comtracting rdms
#     """
#     e1 = np.einsum("pq,qp->", H, gamma1_a + gamma1_b, optimize=True)
#     e2 = 0.5 * np.einsum("pqrs,prqs", V, gamma2_a + gamma2_b, optimize=True)

#     return Ecore + e1 + e2

# def natural_orbitals_from_gamma1(C_mo: np.ndarray, gamma1_mo: np.ndarray):
#     # Symmetrize for numerical stability (MP2 1-RDM may be slightly non-symmetric)
#     g = 0.5 * (gamma1_mo + gamma1_mo.T)

#     # Natural occupations (n) and rotation (U) in the MO basis
#     n, U = np.linalg.eigh(g)  # ascending
#     idx = np.argsort(n)[::-1]  # descending occupations
#     n = n[idx]
#     U = U[:, idx]

#     # AO->NO coefficients
#     C_no = C_mo @ U
#     return C_no, n, U


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
