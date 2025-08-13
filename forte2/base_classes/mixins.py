from dataclasses import dataclass


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
        assert hasattr(
            upstream, "irrep_indices"
        ), "upstream must have an 'irrep_indices' attribute"
        # copy each matrix
        new.C = [arr.copy() for arr in upstream.C]
        new.irrep_indices = upstream.irrep_indices.copy()


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
