from dataclasses import dataclass

import numpy as np

from .mp2_base import MP2Base
from forte2.base_classes import SystemMixin, MOsMixin


@dataclass
class RMP2(MP2Base):
    """
    Density-Fitted Møller-Plesset perturbation theory (DF-MP2) method with RHF canonical orbitals.

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
