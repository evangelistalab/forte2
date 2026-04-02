from dataclasses import dataclass
import time

import numpy as np

from .mp2_base import MP2Base
from .ump2 import UMP2
from forte2.helpers import logger
from forte2.scf import ROHF, UHF


@dataclass
class ROMP2(MP2Base):
    """
    Density-Fitted Møller-Plesset perturbation theory (DF-MP2) method with ROHF canonical orbitals.

    Request optional quantities with the fluent helpers inherited from
    :class:`MP2Base`, for example ``ROMP2().compute_1rdm().compute_2rdm()``.

    Returns
    -------
    float
        MP2 total energy (E_HF + E_corr).
    """

    def __call__(self, parent_method):
        self.parent_method = parent_method
        if not isinstance(parent_method, ROHF):
            raise TypeError("ROMP2 requires an ROHF reference.")
        return self

    def _reference_label(self) -> str:
        return "ROHF"

    def run(self):
        t0 = time.monotonic()
        mem0 = self._memory_snapshot()

        self._startup()

        self._log_rohf_remap()

        # ---- build UHF-like object ----
        uhf_like = self._build_uhf_from_rohf()

        # ---- call UHF MP2 ----
        mp2 = UMP2()
        if self._compute_1rdm:
            mp2.compute_1rdm()
        if self._compute_1rdm_ao:
            mp2.compute_1rdm_ao()
        if self._compute_2rdm:
            mp2.compute_2rdm()
        if self._compute_cumulants:
            mp2.compute_cumulants()
        if self._store_t2:
            mp2.store_t2()
        mp2 = mp2(uhf_like)
        mp2.run()

        # ---- copy results ----
        self.B_iaQ = mp2.B_iaQ
        self.E_corr = mp2.E_corr
        self.E_total = self.parent_method.E + self.E_corr

        if self._compute_1rdm:
            self.gamma1_a = mp2.gamma1_a
            self.gamma1_b = mp2.gamma1_b
            self.gamma1_sf = mp2.gamma1_sf

        if self._compute_1rdm_ao:
            self.gamma1_sf_ao = mp2.gamma1_sf_ao

        if self._compute_2rdm:
            self.gamma2_sf = mp2.gamma2_sf
            self.gamma2_aa = mp2.gamma2_aa
            self.gamma2_ab = mp2.gamma2_ab
            self.gamma2_bb = mp2.gamma2_bb

        if self._compute_cumulants:
            self.lambda2_sf = mp2.lambda2_sf

        if self._store_t2:
            self.t2_a = mp2.t2_a
            self.t2_b = mp2.t2_b
            self.t2_ab = mp2.t2_ab

        self.executed = True
        self._log_completion(time.monotonic() - t0, mp2._t2_norm(), mem0)

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
        return UMP2._build_t2_all(self, B, store_t2=store_t2)

    def _build_uhf_from_rohf(self):
        """
        Build a fake UHF object from ROHF orbitals.
        """

        class FakeUHF(UHF):
            pass

        rohf = self.parent_method
        uhf = object.__new__(FakeUHF)

        C = rohf.C[0]
        eps = rohf.eps[0]

        nd = (rohf.na + rohf.nb) // 2
        ns = rohf.na - nd
        nmo = rohf.nmo

        # ---- orbital coefficients ----
        uhf.C = (C.copy(), C.copy())

        # ---- orbital energies ----
        uhf.eps = (eps.copy(), eps.copy())

        # ---- occupations ----
        occ_a = np.zeros(nmo)
        occ_b = np.zeros(nmo)

        # doubly occupied
        occ_a[:nd] = 1
        occ_b[:nd] = 1

        # singly occupied → alpha only
        occ_a[nd : nd + ns] = 1

        uhf.occ = (occ_a, occ_b)

        # ---- electron counts ----
        uhf.charge = rohf.charge
        uhf.ms = rohf.ms
        uhf.na = rohf.na
        uhf.nb = rohf.nb
        uhf.nmo = rohf.nmo
        uhf.nbf = rohf.nbf
        uhf.executed = True
        uhf.irrep_indices = [rohf.irrep_indices.copy(), rohf.irrep_indices.copy()]
        uhf.irrep_labels = [rohf.irrep_labels.copy(), rohf.irrep_labels.copy()]

        # ---- pass integrals / DF builder ----
        uhf.system = rohf.system
        uhf.fock_builder = rohf.system.fock_builder

        # ---- energies ----
        uhf.E = rohf.E

        return uhf

    def _log_rohf_remap(self):
        logger.log_info1(
            "ROHF reference detected. Mapping ROHF orbitals to a UHF representation and performing UHF-MP2."
        )
