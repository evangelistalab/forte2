from dataclasses import dataclass

from .ump2 import UMP2
from forte2.scf import ROHF, rohf_to_uhf


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
        if not isinstance(parent_method, ROHF):
            raise TypeError("ROMP2 requires an ROHF reference.")
        self._register_parent_method(parent_method)
        return self

    def _reference_label(self) -> str:
        return "ROHF"

    def _startup(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        self._working_reference = rohf_to_uhf(self.parent_method)
        super()._startup(self._working_reference)
