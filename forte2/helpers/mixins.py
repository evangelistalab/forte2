from dataclasses import dataclass, field
import numpy as np


@dataclass
class MOs:
    C: np.ndarray = field(init=False)
