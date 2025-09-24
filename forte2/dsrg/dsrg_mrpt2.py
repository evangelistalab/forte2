from dataclasses import dataclass, field

import numpy as np

from .dsrg_base import DSRGBase

@dataclass
class DSRG_MRPT2(DSRGBase):
    def solve_dsrg(self):
        raise NotImplementedError("DSRG-MRPT2 is not yet implemented.")
    
    def do_reference_relaxation(self):
        raise NotImplementedError("Reference relaxation for DSRG-MRPT2 is not yet implemented.")