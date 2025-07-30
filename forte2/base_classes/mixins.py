from dataclasses import dataclass, field
import numpy as np

from forte2.system.system import System


@dataclass
class MOsMixin:
    """
    Mixin for classes that need to handle molecular orbitals (MOsMixin).
    Contains a list of molecular orbital coefficient matrices.
    """

    @classmethod
    def copy_from_upstream(cls, new, upstream) -> None:
        assert isinstance(new, MOsMixin), "new must be an instance of MOsMixin"
        assert isinstance(
            upstream, MOsMixin
        ), "upstream must be an instance of MOsMixin"
        assert hasattr(upstream, "C"), "upstream must have a 'C' attribute"
        # copy each matrix
        new.C = [arr.copy() for arr in upstream.C]  # uses np.copy here


@dataclass
class SystemMixin:
    """
    Mixin for classes that need to handle a system.
    Contains a reference to the system object.
    """

    @classmethod
    def copy_from_upstream(cls, new, upstream) -> None:
        assert isinstance(new, SystemMixin), "new must be an instance of SystemMixin"
        assert isinstance(
            upstream, SystemMixin
        ), "upstream must be an instance of SystemMixin"
        assert hasattr(upstream, "system"), "upstream must have a 'system' attribute"
        new.system = upstream.system


@dataclass
class MOSpaceMixin:
    """
    Mixin for classes that requires or provides a way to partition molecular orbitals
    into core, active (potentially multiple GASes), and virtual spaces.
    """

    @classmethod
    def copy_from_upstream(cls, new, upstream) -> None:
        assert isinstance(new, MOSpaceMixin), "new must be an instance of MOSpaceMixin"
        assert isinstance(
            upstream, MOSpaceMixin
        ), "upstream must be an instance of MOSpaceMixin"
        assert hasattr(
            upstream, "mo_space"
        ), "upstream must have a 'mo_space' attribute"
        new.mo_space = upstream.mo_space


# @dataclass
# class RDMsMixin:
#     rdms: list[np.ndarray] = field(default_factory=list, init=False)
#     G2: list[np.ndarray] = field(default_factory=list, init=False)
#     G3: list[np.ndarray] = field(default_factory=list, init=False)
