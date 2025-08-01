from dataclasses import dataclass, field
import numpy as np


@dataclass
class _MOSpaceBase:
    """The most general MO space. Essentially used as a utility class for MOSpace and EmbeddingMOSpace."""

    frozen_core_orbitals: list[int]
    core_orbitals: list[list[int]]
    active_orbitals: list[list[int]]
    virtual_orbitals: list[list[int]]
    frozen_virtual_orbitals: list[int]

    def __post_init__(self):
        """
        Initialize the MOSpaceBase and parse the lists of orbitals.
        """
        self._parse_lists()

    def _parse_lists(self):
        self.frozen_core_indices = self.frozen_core_orbitals
        self.nfrozen_core = len(self.frozen_core_orbitals)

        for sublist in self.core_orbitals:
            assert all(
                isinstance(x, int) for x in sublist
            ), "All elements in the sublists must be integers."
        self.ncore_spaces = len(self.core_orbitals)
        self.core_indices = [orb for sublist in self.core_orbitals for orb in sublist]
        self.ncore = sum(len(sublist) for sublist in self.core_orbitals)

        for sublist in self.active_orbitals:
            assert all(
                isinstance(x, int) for x in sublist
            ), "All elements in the sublists must be integers."
        self.ngas = len(self.active_orbitals)
        self.active_indices = [
            orb for sublist in self.active_orbitals for orb in sublist
        ]
        self.nactv = sum(len(sublist) for sublist in self.active_orbitals)

        for sublist in self.virtual_orbitals:
            assert all(
                isinstance(x, int) for x in sublist
            ), "All elements in the sublists must be integers."
        self.nvirt_spaces = len(self.virtual_orbitals)
        self.virtual_indices = [
            orb for sublist in self.virtual_orbitals for orb in sublist
        ]
        self.nvirt = sum(len(sublist) for sublist in self.virtual_orbitals)

        self.frozen_virtual_indices = self.frozen_virtual_orbitals
        self.nfrozen_virtual = len(self.frozen_virtual_orbitals)

        self.nmo = (
            self.nfrozen_core
            + self.ncore
            + self.nactv
            + self.nvirt
            + self.nfrozen_virtual
        )

        # ensure all indices are sorted
        assert (
            sorted(self.frozen_core_orbitals) == self.frozen_core_orbitals
        ), "Frozen core orbitals must be sorted."
        assert all(
            sorted(sublist) == sublist for sublist in self.core_orbitals
        ), "All core orbitals must be sorted lists of integers."
        assert all(
            sorted(sublist) == sublist for sublist in self.active_orbitals
        ), "All active orbitals must be sorted lists of integers."
        assert all(
            sorted(sublist) == sublist for sublist in self.virtual_orbitals
        ), "All virtual orbitals must be sorted lists of integers."
        assert (
            sorted(self.frozen_virtual_orbitals) == self.frozen_virtual_orbitals
        ), "Frozen virtual orbitals must be sorted."

        # ensure no indices are repeated
        assert (
            len(set(self.frozen_core_indices)) == self.nfrozen_core
        ), "Frozen core indices must be unique."
        assert len(set(self.core_indices)) == self.ncore, "Core indices must be unique."
        assert (
            len(set(self.active_indices)) == self.nactv
        ), "Active indices must be unique."
        assert (
            len(set(self.frozen_virtual_indices)) == self.nfrozen_virtual
        ), "Frozen virtual indices must be unique."

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
        self.core = []
        i = self.frozen_core.stop
        for core in self.core_orbitals:
            self.core.append(slice(i, i + len(core)))
            i += len(core)
        self.actv = []
        i = self.core[-1].stop
        for actv in self.active_orbitals:
            self.actv.append(slice(i, i + len(actv)))
            i += len(actv)
        self.virt = []
        i = self.actv[-1].stop
        for virt in self.virtual_orbitals:
            self.virt.append(slice(i, i + len(virt)))
            i += len(virt)
        self.frozen_virt = slice(self.virt[-1].stop, self.nmo)


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
        self.active_orbitals = list(range(nfc + nc, nfc + nc + na))
        self.frozen_virtual_orbitals = list(range(self.nmo - nfv, self.nmo))

    def _infer_virtual_indices(self):
        assert isinstance(self.core_orbitals, list), "core_orbitals must be a list."
        assert isinstance(
            self.frozen_core_orbitals, list
        ), "frozen_core_orbitals must be a list."
        assert isinstance(
            self.frozen_virtual_orbitals, list
        ), "frozen_virtual_orbitals must be a list."

        _active_flat = [orb for sublist in self.active_orbitals for orb in sublist]

        all_indices = list(range(self.nmo))
        virtual_indices = sorted(
            list(
                set(all_indices)
                - set(self.core_orbitals)
                - set(self.frozen_core_orbitals)
                - set(_active_flat)
                - set(self.frozen_virtual_orbitals),
            )
        )
        if len(virtual_indices) < 0:
            raise ValueError(
                f"The sum of frozen_core, core, active, and frozen_virtual dimensions ({len(self.frozen_core_orbitals) + len(self.core_orbitals) + len(_active_flat) + len(self.frozen_virtual_orbitals)}) exceeds the total number of orbitals ({self.nmo})."
            )
        return virtual_indices

    def _parse_lists(self):
        if all(isinstance(x, int) for x in self.active_orbitals):
            self.active_orbitals = [self.active_orbitals]

        self.virtual_indices = self._infer_virtual_indices()

        _mo_space = _MOSpaceBase(
            frozen_core_orbitals=self.frozen_core_orbitals,
            core_orbitals=[self.core_orbitals],
            active_orbitals=self.active_orbitals,
            virtual_orbitals=[self.virtual_indices],
            frozen_virtual_orbitals=self.frozen_virtual_orbitals,
        )

        self.ngas = _mo_space.ngas

        self.nfrozen_core = _mo_space.nfrozen_core
        self.frozen_core_indices = _mo_space.frozen_core_indices

        self.ncore = _mo_space.ncore
        self.core_indices = _mo_space.core_indices

        self.docc_orbitals = self.docc_indices = (
            self.frozen_core_indices + self.core_indices
        )

        self.nactv = _mo_space.nactv
        self.active_indices = _mo_space.active_indices

        self.nvirt = _mo_space.nvirt
        self.virtual_indices = _mo_space.virtual_indices

        self.nfrozen_virtual = _mo_space.nfrozen_virtual
        self.frozen_virtual_indices = _mo_space.frozen_virtual_indices

        self.uocc_orbitals = self.uocc_indices = (
            self.virtual_indices + self.frozen_virtual_indices
        )

        if self.ncore + self.nactv == 0:
            raise ValueError(
                "Neither core nor active orbitals are defined. There will be no electrons to correlate."
            )

        # slices for the different spaces in the contiguous full space
        self.frozen_core = _mo_space.frozen_core
        self.core = _mo_space.core[0]
        self.docc = slice(0, self.core.stop)  # both core and frozen core
        self.actv = slice(_mo_space.actv[0].start, _mo_space.actv[-1].stop)
        self.virt = _mo_space.virt[0]
        self.frozen_virt = _mo_space.frozen_virt
        self.uocc = slice(self.virt.start, self.nmo)  # both virtual and frozen virtual
        self.gas = _mo_space.actv

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

        self.contig_to_orig = _mo_space.contig_to_orig
        self.orig_to_contig = _mo_space.orig_to_contig


@dataclass
class EmbeddingMOSpace:
    """Simplified attribute list as this is only used for semicanonicalization."""

    nmo: int
    frozen_core_orbitals: list[int]
    B_core_orbitals: list[int]
    A_core_orbitals: list[int]
    active_orbitals: list[int]
    A_virtual_orbitals: list[int]
    B_virtual_orbitals: list[int]
    frozen_virtual_orbitals: list[int]

    def __post_init__(self):
        _mo_space = _MOSpaceBase(
            frozen_core_orbitals=self.frozen_core_orbitals,
            core_orbitals=[self.B_core_orbitals, self.A_core_orbitals],
            active_orbitals=[self.active_orbitals],
            virtual_orbitals=[self.A_virtual_orbitals, self.B_virtual_orbitals],
            frozen_virtual_orbitals=self.frozen_virtual_orbitals,
        )

        self.frozen_core = _mo_space.frozen_core
        self.B_core = _mo_space.core[0]
        self.A_core = _mo_space.core[1]
        self.core = slice(0, self.A_core.stop)
        self.actv = _mo_space.actv[0]
        self.A_virt = _mo_space.virt[0]
        self.B_virt = _mo_space.virt[1]
        self.frozen_virt = _mo_space.frozen_virt
        self.contig_to_orig = _mo_space.contig_to_orig
        self.orig_to_contig = _mo_space.orig_to_contig
