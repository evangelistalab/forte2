from dataclasses import dataclass, field
import numpy as np

from forte2 import RelSlaterRules, hilbert_space
from forte2.state import State
from forte2.base_classes import SystemMixin, MOsMixin
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.jkbuilder import SpinorbitalIntegrals


@dataclass
class RelCI(SystemMixin, MOsMixin):
    state: State
    active_spinorbitals: list[int]
    core_spinorbitals: list[int] = field(default_factory=list)

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        if not self.system.two_component:
            self.C = convert_coeff_spatial_to_spinor(self.system, self.C)
            self.system.two_component = True

        ints = SpinorbitalIntegrals(
            self.system,
            self.C[0],
            self.active_spinorbitals,
            self.core_spinorbitals,
            use_aux_corr=True,
        )

        nspinor = len(self.active_spinorbitals)
        ncore = len(self.core_spinorbitals)

        slater_rules = RelSlaterRules(
            nspinor=nspinor,
            scalar_energy=ints.E.real,
            one_electron_integrals=ints.H,
            two_electron_integrals=ints.V,
        )

        nel = self.state.na + self.state.nb - ncore
        dets = hilbert_space(nspinor, nel, 0)
        H = np.zeros((len(dets),) * 2, dtype=complex)
        for i in range(len(dets)):
            for j in range(i + 1):
                H[i, j] = slater_rules.slater_rules(dets[i], dets[j])
                H[j, i] = np.conj(H[i, j])

        self.evals, self.evecs = np.linalg.eigh(H)
        return self
