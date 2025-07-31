from dataclasses import dataclass, field
import numpy as np


@dataclass
class MOSpace:
    """
    A class to store the partitioning of the molecular orbital space.

    Parameters
    ----------
    nmo : int
        The total number of molecular orbitals.
    active_orbitals : list[int] | list[list[int]]
        A list of integers or a list of lists of integers storing the orbital indices of the GASes.
        If a single list of integers is provided, it is treated as the CAS (a single GAS).
    core_orbitals : list[int], optional, default=[]
        A list of integers storing the core orbital indices.
    frozen_core_orbitals : list[int], optional, default=[]
        A list of integers storing the frozen core orbital indices.
    frozen_virtual_orbitals : list[int], optional, default=[]
        A list of integers storing the frozen virtual orbital indices.

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

    nmo: int
    active_orbitals: list[int] | list[list[int]] = field(default_factory=list)
    core_orbitals: list[int] = field(default_factory=list)
    frozen_core_orbitals: list[int] = field(default_factory=list)
    frozen_virtual_orbitals: list[int] = field(default_factory=list)

    def __post_init__(self):
        ints_as_args = any(
            [
                isinstance(self.active_orbitals, int),
                isinstance(self.core_orbitals, int),
                isinstance(self.frozen_core_orbitals, int),
                isinstance(self.frozen_virtual_orbitals, int),
            ]
        )
        if ints_as_args:
            self._convert_integer_arguments_to_lists()

        self._parse_lists()

    def _convert_integer_arguments_to_lists(self):
        def _to_int(x):
            if isinstance(x, int):
                return x
            elif isinstance(x, list) and len(x) == 0:
                return 0
            else:
                raise ValueError(
                    "If one of the parameters is an integer, all must be integers or not provided."
                )

        nfc = _to_int(self.frozen_core_orbitals)
        nc = _to_int(self.core_orbitals)
        na = _to_int(self.active_orbitals)
        nfv = _to_int(self.frozen_virtual_orbitals)
        nv = self.nmo - (nfc + nc + na + nfv)
        assert (
            nv >= 0
        ), f"The sum of frozen_core, core, active, and frozen_virtual dimensions ({nfc + nc + na + nfv}) exceeds the total number of orbitals ({self.nmo})."
        self.frozen_core_orbitals = list(range(nfc))
        self.core_orbitals = list(range(nfc, nfc + nc))
        self.active_orbitals = [list(range(nfc + nc, nfc + nc + na))]
        self.frozen_virtual_orbitals = list(range(self.nmo - nfv, self.nmo))

    def _parse_lists(self):
        # Validate input types
        assert isinstance(self.active_orbitals, list), "active_orbitals must be a list."
        assert isinstance(self.core_orbitals, list), "core_orbitals must be a list."
        assert isinstance(
            self.frozen_core_orbitals, list
        ), "frozen_core_orbitals must be a list."
        assert isinstance(
            self.frozen_virtual_orbitals, list
        ), "frozen_virtual_orbitals must be a list."

        # Validate elements of active_orbitals
        if all(isinstance(x, int) for x in self.active_orbitals):
            self.ngas = 1
            self.active_orbitals = [self.active_orbitals]
        elif all(isinstance(x, list) for x in self.active_orbitals):
            for sublist in self.active_orbitals:
                assert all(
                    isinstance(x, int) for x in sublist
                ), "All elements in the sublists must be integers."
            self.ngas = len(self.active_orbitals)

        # ensure all indices are sorted
        assert all(
            sorted(sublist) == sublist for sublist in self.active_orbitals
        ), "All active orbitals must be sorted lists of integers."
        assert (
            sorted(self.core_orbitals) == self.core_orbitals
        ), "Core orbitals must be sorted."
        assert (
            sorted(self.frozen_core_orbitals) == self.frozen_core_orbitals
        ), "Frozen core orbitals must be sorted."
        assert (
            sorted(self.frozen_virtual_orbitals) == self.frozen_virtual_orbitals
        ), "Frozen virtual orbitals must be sorted."

        # store flattened lists ('*_indices') of all orbitals
        self.active_indices = [
            orb for sublist in self.active_orbitals for orb in sublist
        ]
        self.core_indices = self.core_orbitals
        self.frozen_core_indices = self.frozen_core_orbitals
        self.frozen_virtual_indices = self.frozen_virtual_orbitals
        self.docc_indices = self.docc_orbitals = (
            self.frozen_core_orbitals + self.core_orbitals
        )

        # store the number of orbitals in each space
        self.nactv = sum(len(sublist) for sublist in self.active_orbitals)
        self.ncore = len(self.core_orbitals)
        self.nfrozen_core = len(self.frozen_core_orbitals)
        self.nfrozen_virtual = len(self.frozen_virtual_orbitals)

        # sanity check on total number of orbitals
        ndef = self.nfrozen_core + self.ncore + self.nactv + self.nfrozen_virtual
        if ndef > self.nmo:
            raise ValueError(
                f"The sum of frozen_core, core, active, and frozen_virtual dimensions ({ndef}) exceeds the total number of orbitals ({self.nmo})."
            )

        if self.ncore + self.nactv == 0:
            raise ValueError(
                "Neither core nor active orbitals are defined. There will be no electrons to correlate."
            )

        # infer virtual indices
        all_indices = list(range(self.nmo))
        self.virtual_indices = sorted(
            list(
                set(all_indices)
                - set(self.active_indices)
                - set(self.core_indices)
                - set(self.frozen_core_indices)
                - set(self.frozen_virtual_indices)
            )
        )
        self.nvirt = len(self.virtual_indices)

        # ensure no indices are repeated
        assert (
            len(set(self.active_indices)) == self.nactv
        ), "Active orbitals must be unique."
        assert (
            len(set(self.core_orbitals)) == self.ncore
        ), "Core orbitals must be unique."
        assert (
            len(set(self.frozen_core_orbitals)) == self.nfrozen_core
        ), "Frozen core orbitals must be unique."
        assert (
            len(set(self.frozen_virtual_orbitals)) == self.nfrozen_virtual
        ), "Frozen virtual orbitals must be unique."

        assert (
            len(
                set(
                    self.active_indices
                    + self.core_indices
                    + self.frozen_core_indices
                    + self.frozen_virtual_indices
                )
            )
            == self.nactv + self.ncore + self.nfrozen_core + self.nfrozen_virtual
        ), "All orbital indices must be unique across active, core, frozen core, and frozen virtual spaces."

        # permutation array that makes spaces contiguous:
        # [frozen_core, core, gas1, gas2, ..., virt, frozen_virtual]
        # such that C_contig = C[:, self.contig_to_orig]
        # and C_orig = C_contig[:, self.orig_to_contig]
        self.contig_to_orig = np.argsort(
            self.frozen_core_indices
            + self.core_indices
            + self.active_indices
            + self.virtual_indices
            + self.frozen_virtual_indices,
        )
        self.orig_to_contig = np.zeros_like(self.contig_to_orig, dtype=int)
        self.orig_to_contig[self.contig_to_orig] = np.arange(self.nmo, dtype=int)

        # slices for the different spaces in the contiguous full space
        self.frozen_core = slice(0, self.nfrozen_core)
        self.core = slice(self.frozen_core.stop, self.frozen_core.stop + self.ncore)
        self.docc = slice(0, self.core.stop)  # both core and frozen core
        self.actv = slice(self.core.stop, self.core.stop + self.nactv)
        self.virt = slice(self.actv.stop, self.actv.stop + self.nvirt)
        self.frozen_virt = slice(self.virt.stop, self.nmo)
        self.uocc = slice(self.virt.start, self.nmo)  # both virtual and frozen virtual
        self.gas = []
        i = self.core.stop
        for actv in self.active_orbitals:
            self.gas.append(slice(i, i + len(actv)))
            i += len(actv)

        # slice to get the correlated space from the full space
        self.corr = slice(self.core.start, self.virt.stop)

        # slices in the correlated space
        self.core_corr = slice(0, self.ncore)
        self.actv_corr = slice(self.ncore, self.ncore + self.nactv)
        self.virt_corr = slice(self.actv_corr.stop, self.actv_corr.stop + self.nvirt)
        self.gas_corr = []
        i = self.ncore
        for actv in self.active_orbitals:
            self.gas_corr.append(slice(i, i + len(actv)))
            i += len(actv)
