from dataclasses import dataclass

from forte2.base_classes import ActiveSpaceSolver, SystemMixin, MOsMixin
from forte2.x2c import get_hcore_x2c
from forte2.helpers.matrix_functions import block_diag_2x2
from .relci import RelCI


@dataclass
class SISO(SystemMixin, MOsMixin):
    snso_type: str = "dcb"

    def __call__(self, parent_method):
        assert isinstance(
            parent_method, ActiveSpaceSolver
        ), "parent_method must be an instance of ActiveSpaceSolver"
        assert (
            parent_method.system.x2c_type == "sf"
        ), "SISO requires the X2C type to be 'sf'."
        self.parent_method = parent_method

    def run(self):
        if not self.parent_method.executed:
            self.parent_method.run()

        SystemMixin.copy_from_upstream(self, self.parent_method)
        MOsMixin.copy_from_upstream(self, self.parent_method)
        assert (
            not self.system.two_component
        ), "SISO requires a one-component upstream method."

        h_x2cso = get_hcore_x2c(self.system, x2c_type="so", snso_type=self.snso_type)
        h_sd = h_x2cso - block_diag_2x2(get_hcore_x2c(self.system, x2c_type="sf"))

        state = self.parent_method.state
        ci = RelCI(state)

        return self
