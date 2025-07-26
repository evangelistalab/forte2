from dataclasses import dataclass, field
import numpy as np


@dataclass
class MOSpace:
    """
    A class to store the partitioning of the molecular orbital space.

    Parameters
    ----------
    active_orbitals : list[int] | list[list[int]]
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
    active_indices : list[int]
        A flattened list of all active orbital indices across all GASes.
    core_indices : list[int]
        A list of core orbital indices, same as core_orbitals.
    """

    active_orbitals: list[int] | list[list[int]]
    core_orbitals: list[int] = field(default_factory=list)

    def __post_init__(self):
        assert isinstance(self.active_orbitals, list), "active_spaces must be a list."
        if all(isinstance(x, int) for x in self.active_orbitals):
            self.ngas = 1
            self.active_orbitals = [self.active_orbitals]
        elif all(isinstance(x, list) for x in self.active_orbitals):
            for sublist in self.active_orbitals:
                assert all(
                    isinstance(x, int) for x in sublist
                ), "All elements in the sublists must be integers."
            self.ngas = len(self.active_orbitals)
        assert all(
            sorted(sublist) == sublist for sublist in self.active_orbitals
        ), "All active spaces must be sorted lists of integers."

        self.nactv = sum(len(sublist) for sublist in self.active_orbitals)
        self.ncore = len(self.core_orbitals)
        assert (
            sorted(self.core_orbitals) == self.core_orbitals
        ), "Core orbitals must be sorted."
        # store a flattened list of all active orbitals
        self.active_indices = [
            orb for sublist in self.active_orbitals for orb in sublist
        ]
        self.core_indices = self.core_orbitals
        assert (
            sorted(self.active_indices) == self.active_indices
        ), "Active orbitals must be sorted."
        assert (
            len(set(self.active_indices)) == self.nactv
        ), "Active orbitals must be unique."
        assert (
            len(set(self.core_orbitals)) == self.ncore
        ), "Core orbitals must be unique."

        assert (
            len(set(self.active_indices + self.core_orbitals))
            == self.nactv + self.ncore
        ), "Active and core orbitals must not overlap."

    def compute_contiguous_permutation(self, nmo):
        """
        Compute the permutation of orbitals to make the core, active, and virtual spaces
        contiguous in the flattened orbital array.
        """
        core = self.core_orbitals
        actv_flat = [item for sublist in self.active_orbitals for item in sublist]
        virt = sorted(list(set(range(nmo)) - set(core) - set(actv_flat)))
        self.contig_to_orig = np.argsort(core + actv_flat + virt)
        self.orig_to_contig = np.zeros_like(self.contig_to_orig, dtype=int)
        self.orig_to_contig[self.contig_to_orig] = np.arange(nmo, dtype=int)
        self.core = slice(0, len(core))
        self.virt = slice(len(core) + len(actv_flat), nmo)
        self.actv = []
        i = len(core)
        for actv in self.active_orbitals:
            self.actv.append(slice(i, i + len(actv)))
            i += len(actv)
        if self.ngas == 1:
            self.actv = self.actv[0]
