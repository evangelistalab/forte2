from dataclasses import dataclass, field
import numpy as np

from forte2.system.system import System


@dataclass
class MOsMixin:
    """
    Mixin for classes that need to handle molecular orbitals (MOsMixin).
    Contains a list of molecular orbital coefficient matrices.
    """

    # C is a list of numpy arrays, each representing the coefficients of a molecular orbital
    C: list[np.ndarray] = field(default_factory=list, init=False)

    @classmethod
    def copy_from_upstream(cls, new, upstream) -> None:
        # copy each matrix
        new.C = [arr.copy() for arr in upstream.C]  # uses np.copy here


@dataclass
class SystemMixin:
    """
    Mixin for classes that need to handle a system.
    Contains a reference to the system object.
    """

    system: System = field(default=None, init=False)

    @classmethod
    def copy_from_upstream(cls, new, upstream) -> None:
        new.system = upstream.system


# @dataclass
# class RDMsMixin:
#     rdms: list[np.ndarray] = field(default_factory=list, init=False)
#     G2: list[np.ndarray] = field(default_factory=list, init=False)
#     G3: list[np.ndarray] = field(default_factory=list, init=False)
