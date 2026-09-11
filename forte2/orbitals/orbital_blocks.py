from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _slice_indices(sl: slice) -> NDArray:
    """Return the orbital indices covered by a contiguous MO space slice."""
    return np.arange(sl.start, sl.stop)


def _blocks_by_labels(sl: slice, labels: NDArray) -> list[NDArray]:
    """Split a contiguous MO space slice into blocks with indices having the same labels."""
    idx = _slice_indices(sl)
    if idx.size == 0:
        return [idx]

    block_labels = labels[idx]
    return [idx[block_labels == label] for label in range(int(labels.max()) + 1)]


class OrbitalBlockBuilder:
    """
    Build orbital index blocks that preserve requested orbital structure.

    The returned indices are in contiguous MO ordering unless noted otherwise.
    If irrep labels are provided, each requested slice is split into irrep
    subblocks. The GAS structure is preserved by applying this splitting
    separately to each GAS slice.

    Parameters
    ----------
    mo_space : forte2.MOSpace | forte2.EmbeddingMOSpace
        MO-space partition. Named spaces are resolved as attributes of this object.
    irrep_indices : np.ndarray or list[int], optional
        Orbital irrep labels in contiguous MO ordering.
    """

    def __init__(
        self,
        mo_space,
        irrep_indices: ArrayLike | None = None,
    ) -> None:
        if mo_space is None:
            raise ValueError("mo_space is required.")
        if not hasattr(mo_space, "nmo"):
            raise ValueError("mo_space must define nmo.")

        self.mo_space = mo_space
        self.irrep_indices = (
            None if irrep_indices is None else np.asarray(irrep_indices, dtype=int)
        )
        if self.irrep_indices is not None and self.irrep_indices.shape != (
            self.mo_space.nmo,
        ):
            raise ValueError("irrep_indices must have one entry per MO.")
        if self.irrep_indices is not None and np.any(self.irrep_indices < 0):
            raise ValueError("irrep_indices must be non-negative.")

    def blocks_for_slice(self, sl: slice) -> list[NDArray]:
        """
        Return independent rotation blocks for a contiguous MO space slice.

        Parameters
        ----------
        sl : slice
            Slice in contiguous MO ordering.

        Returns
        -------
        list[np.ndarray]
            Contiguous-order index arrays. If symmetry is available, blocks are
            split by irrep; otherwise the full slice is returned as one block.
        """
        if sl.start is None or sl.stop is None or sl.step not in (None, 1):
            raise ValueError(
                "Orbital slices must have explicit start/stop and unit step."
            )

        if not 0 <= sl.start <= sl.stop <= self.mo_space.nmo:
            raise ValueError("Orbital slice is outside the MO space.")

        idx = _slice_indices(sl)
        if self.irrep_indices is None or idx.size == 0:
            return [idx]

        return _blocks_by_labels(sl, self.irrep_indices)

    def blocks_for_space(self, label: str) -> list[NDArray]:
        """
        Return independent rotation blocks for an orbital space.

        Parameters
        ----------
        name : str
            Name of an attribute on ``mo_space`` that resolves to a slice or a
            list of slices.

        Returns
        -------
        list[np.ndarray]
            Contiguous-order orbital index blocks.
        """
        if not isinstance(label, str):
            raise TypeError("Orbital space names must be strings.")
        if not hasattr(self.mo_space, label):
            raise ValueError(f"Unknown orbital space: {label!r}.")

        value = getattr(self.mo_space, label)
        slices = value if isinstance(value, list) else [value]
        if not all(isinstance(sl, slice) for sl in slices):
            raise TypeError(
                f"Orbital space {label!r} does not resolve to slice objects."
            )

        return [block for sl in slices for block in self.blocks_for_slice(sl)]

    def blocks_for_spaces(self, labels: Iterable[str]) -> list[NDArray]:
        """
        Return independent rotation blocks for named orbital spaces.

        Parameters
        ----------
        names : iterable[str]
            Names of ``mo_space`` attributes that resolve to slices.

        Returns
        -------
        list[np.ndarray]
            Contiguous-order orbital index blocks.
        """
        if isinstance(labels, str):
            raise TypeError("Use blocks_for_space() for a single orbital space.")
        return [block for label in labels for block in self.blocks_for_space(label)]

    def active_blocks(self, relative_index: bool = True) -> list[NDArray]:
        """
        Return active-space blocks preserving GAS and symmetry structure.

        Parameters
        ----------
        relative : bool, optional, default=True
            If True, return indices relative to the start of the active space.
            If False, return full contiguous MO indices.

        Returns
        -------
        list[np.ndarray]
            Active orbital index blocks. GAS slices are kept separate, and each
            GAS is split further by irrep when irrep labels are available.
        """
        name = "gas" if hasattr(self.mo_space, "gas") else "actv"
        blocks = self.blocks_for_space(name)
        if relative_index:
            offset = self.mo_space.actv.start
            blocks = [block - offset for block in blocks]
        return blocks
