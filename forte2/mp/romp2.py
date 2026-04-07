from dataclasses import dataclass
import time

import numpy as np

from .mp2_base import MP2Base
from .ump2 import UMP2
from forte2.helpers import logger
from forte2.scf import ROHF, UHF, rohf_to_uhf


@dataclass
class ROMP2(UMP2):
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

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        self.parent_method = rohf_to_uhf(self.parent_method)
        super()._startup()

    # def _log_rohf_remap(self):
    #     logger.log_info1(
    #         "ROHF reference detected. Mapping ROHF orbitals to a UHF representation and performing UHF-MP2."
    #     )
