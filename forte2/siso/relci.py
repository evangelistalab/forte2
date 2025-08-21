from dataclasses import dataclass

from forte2.base_classes import SystemMixin, MOsMixin
from forte2.scf.hf import convert_coeff_spatial_to_spinor


@dataclass
class RelCI(SystemMixin, MOsMixin):
    param1: int
    param2: float
    param3: str

    def __call__(self, parent_method):
        self.parent_method = parent_method
        return self

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        if not self.system.two_component:
            self.C = convert_coeff_spatial_to_spinor(self.C)
            self.system.two_component = True
        
        ints = RestrictedMOIntegrals(
            self.system,
            self.C[0],
            self.active_indices,
            self.core_indices,
            use_aux_corr=True,
        )
