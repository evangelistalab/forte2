from dataclasses import dataclass
import time

import numpy as np

from .mp2_base import MP2Base


@dataclass
class ROMP2(MP2Base):
    """
    Density-Fitted Møller-Plesset perturbation theory (DF-MP2) method with ROHF canonical orbitals.

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

    def _reference_label(self) -> str:
        return "ROHF"

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

        # ---- custom RDM pipeline ----
        self._initialize_rdm_outputs()
        need_gamma1 = (
            self.compute_1rdm
            or self.compute_1rdm_ao
            or self.compute_2rdm
            or self.compute_cumulants
        )

        if need_gamma1:
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

        self.executed = True
        self._log_completion(time.monotonic() - t0, self._t2_norm(), mem0)
        return self.E_total

    def _startup(self):
        self._copy_restricted_reference()

        self.docc = (self.parent_method.na + self.parent_method.nb) // 2
        self.socc = self.parent_method.na - self.docc
        self.nocc = self.docc + self.socc
        self.nvir = self.parent_method.nuocc

    def _build_df_iaQ(self):
        return self._build_restricted_df_iaQ(self.nocc)

    def _build_t2_all(self, B, store_t2=True):
        nd = self.docc
        ns = self.socc
        nvir = self.nvir

        eps = self.eps
        eps_d = eps[:nd]
        eps_s = eps[nd : nd + ns]
        eps_v = eps[nd + ns :]

        E_corr = 0.0
        t2 = np.empty((self.nocc, self.nocc, nvir, nvir)) if store_t2 else None
        t2_as = np.empty_like(t2) if store_t2 else None

        # doubly-doubly contribution
        for i in range(nd):
            Bi = B[i]  # (nvir, naux)
            for j in range(nd):
                Bj = B[j]
                gijab = Bi @ Bj.T  # (a,b)
                denom = eps_d[i] + eps_d[j] - eps_v[:, None] - eps_v[None, :]
                tijab = self._safe_divide(gijab, denom)

                if store_t2:
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

                if store_t2:
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

                if store_t2:
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
