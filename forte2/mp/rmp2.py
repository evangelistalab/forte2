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
        self._copy_restricted_reference()
        self.nocc = self.parent_method.na
        self.nvir = self.parent_method.nuocc

    def _build_df_iaQ(self):
        """
        Build 3-index integrals (ia|Q) using density fitting.

        Returns:
        --------
            B_iaQ: 3-index integrals (ia|Q) as a numpy array of shape (n_occ, n_vir, n_aux).
        """
        return self._build_restricted_df_iaQ(self.nocc)

    def _build_t2_all(self, B, store_t2=True):
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
        t2 = np.empty((nocc, nocc, nvir, nvir)) if store_t2 else None
        # antisym only in (a,b): t̃ = 2t - t^{ba}
        t2_as = np.empty_like(t2) if store_t2 else None

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
                if store_t2:
                    t2[i, j] = tijab
                    t2_as[i, j] = 2.0 * tijab - tijab.T

                # energy contribution
                E_corr += np.sum((2.0 * gijab - gijab.T) * tijab)

        return t2, t2_as, E_corr
