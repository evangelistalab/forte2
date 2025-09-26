from dataclasses import dataclass, field

import numpy as np

from forte2.jkbuilder.mointegrals import RestrictedMOIntegrals, SpinorbitalIntegrals
from .dsrg_base import DSRGBase


@dataclass
class DSRG_MRPT2(DSRGBase):
    def get_integrals(self):
        if self.parent_method.final_orbital != "semicanonical":
            # TODO: semi-canonicalize first
            raise NotImplementedError("Only semicanonical orbitals are implemented.")

        if self.two_component:
            ints = SpinorbitalIntegrals(
                system=self.system,
                C=self._C,
                spinorbitals=list(
                    range(self.mo_space.coor.start, self.mo_space.coor.stop)
                ),
                core_spinorbitals=list(range(0, self.mo_space.frozen_core.stop)),
                antisymmetrize=True,
            )
            cumulants = dict()
            cumulants["gamma1"] = self.parent_method.make_average_1rdm()
            cumulants["eta1"] = (
                np.eye(cumulants["gamma1"].shape[0], dtype=complex)
                - cumulants["gamma1"]
            )
            cumulants["lambda2"] = self.parent_method.make_average_2cumulant()
            cumulants["lambda3"] = self.parent_method.make_average_3cumulant()
            return ints, cumulants
        else:
            raise NotImplementedError("Only two-component integrals are implemented.")

    def solve_dsrg(self):
        raise NotImplementedError("DSRG-MRPT2 is not yet implemented.")

    def do_reference_relaxation(self):
        raise NotImplementedError(
            "Reference relaxation for DSRG-MRPT2 is not yet implemented."
        )
