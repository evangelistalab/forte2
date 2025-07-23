from dataclasses import dataclass, field
import numpy as np


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

        assert (
            len(set(self.active_orbitals + self.core_orbitals))
            == self.nactv + self.ncore
        ), "Active and core orbitals must not overlap."

    def make_spaces_contiguous(self, nmo):
        """
        Swap the orbitals to ensure that the core, active, and virtual orbitals
        are contiguous in the flattened orbital array.
        """
        core = self.core_orbitals
        actv_sorted = [sorted(actv) for actv in self.active_spaces]
        actv_sorted_flat = [item for sublist in actv_sorted for item in sublist]
        virt = sorted(list(set(range(nmo)) - set(core) - set(actv_sorted_flat)))
        self.argsort = np.argsort(core + actv_sorted_flat + virt)
        self.inv_argsort = actv_sorted_flat
        self.core = slice(0, len(core))
        self.virt = slice(len(core) + len(actv_sorted_flat), nmo)
        self.actv = []
        i = len(core)
        for actv in actv_sorted:
            self.actv.append(slice(i, i + len(actv)))
            i += len(actv)
        if len(actv_sorted) == 1:
            self.actv = self.actv[0]
