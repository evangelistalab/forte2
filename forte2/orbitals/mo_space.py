from dataclasses import dataclass, field


@dataclass
class MOSpace:
    """
    A class to store the partitioning of the molecular orbital space.

    Parameters
    ----------
    active_spaces : list[int] | list[list[int]]
        A list of integers or a list of lists of integers storing the orbital indices of the GASes.
        If a single list of integers is provided, it is treated as the CAS (a single GAS).
    core_orbitals : list[int], optional, default=[]
        A list of integers storing the core orbital indices.

    Attributes
    ----------
    ngas : int
        The number of GASes (General Active Spaces) defined by the active_spaces.
    nactv : int
        The total number of active orbitals across all GASes.
    ncore : int
        The number of core orbitals.
    active_orbitals : list[int]
        A flattened list of all active orbitals across all GASes.
    """

    active_spaces: list[int] | list[list[int]]
    core_orbitals: list[int] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.active_spaces, list), "active_spaces must be a list."
        if all(isinstance(x, int) for x in self.active_spaces):
            self.ngas = 1
            self.active_spaces = [self.active_spaces]
        elif all(isinstance(x, list) for x in self.active_spaces):
            for sublist in self.active_spaces:
                assert all(
                    isinstance(x, int) for x in sublist
                ), "All elements in the sublists must be integers."
            self.ngas = len(self.active_spaces)

        self.nactv = sum(len(sublist) for sublist in self.active_spaces)
        self.ncore = len(self.core_orbitals)
        # store a flattened list of all active orbitals
        self.active_orbitals = [
            orb for sublist in self.active_spaces for orb in sublist
        ]
        assert (
            len(set(self.active_orbitals)) == self.nactv
        ), "Active orbitals must be unique."
        assert (
            len(set(self.core_orbitals)) == self.ncore
        ), "Core orbitals must be unique."
