from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from forte2.state.mo_space import blocks_by_labels, slice_indices

OrbitalSpace: TypeAlias = str | slice


def _normalize_spaces(
    spaces: OrbitalSpace | Iterable[OrbitalSpace] | None,
) -> list[OrbitalSpace] | None:
    if spaces is None:
        return None
    if isinstance(spaces, (str, slice)):
        return [spaces]
    return list(spaces)


class OrbitalBlockBuilder:
    """
    Build orbital index blocks that preserve requested orbital structure.

    The returned indices are in contiguous MO ordering unless noted otherwise.
    If irrep labels are provided for a non-C1 system, each requested slice is
    split into irrep-homogeneous blocks. GAS structure is preserved by applying
    this splitting separately to each GAS slice.

    Parameters
    ----------
    system : forte2.System
        System object used to determine whether point-group symmetry is active.
    mo_space : forte2.MOSpace | forte2.EmbeddingMOSpace
        MO-space partition. Named spaces are resolved as attributes of this object.
    irrep_indices : np.ndarray or list[int], optional
        Orbital irrep labels in contiguous MO ordering.
    spaces : list[str | slice], optional
        Default orbital spaces to use when ``blocks_for_spaces`` is called without
        an explicit list. Strings name ``mo_space`` attributes, for example
        ``"core"``, ``"gas"``, ``"actv"``, or ``"virt"``. A name that resolves to
        a list of slices, such as ``"gas"``, is expanded while preserving the list
        order.
    """

    def __init__(
        self,
        system,
        mo_space,
        irrep_indices: ArrayLike | None = None,
        spaces: OrbitalSpace | Iterable[OrbitalSpace] | None = None,
    ) -> None:
        if mo_space is None:
            raise ValueError("mo_space is required.")
        if not hasattr(mo_space, "nmo"):
            raise ValueError("mo_space must define nmo.")

        self.system = system
        self.mo_space = mo_space
        self.spaces = _normalize_spaces(spaces)
        self.irrep_indices = (
            None if irrep_indices is None else np.asarray(irrep_indices, dtype=int)
        )
        if self.irrep_indices is not None and self.irrep_indices.shape != (
            self.mo_space.nmo,
        ):
            raise ValueError("irrep_indices must have one entry per MO.")

    def slices_for_space(self, space: OrbitalSpace) -> list[slice]:
        """
        Resolve a named orbital space to one or more contiguous MO slices.

        Parameters
        ----------
        space : str or slice
            A string naming an attribute on ``mo_space`` or a slice in contiguous
            MO ordering. Names that resolve to lists of slices, such as ``"gas"``,
            are expanded.

        Returns
        -------
        list[slice]
            Contiguous MO-space slices.
        """
        if isinstance(space, slice):
            return [space]

        if not isinstance(space, str):
            raise TypeError("Orbital spaces must be specified as strings or slices.")

        if not hasattr(self.mo_space, space):
            raise ValueError(f"Unknown orbital space: {space!r}.")

        value = getattr(self.mo_space, space)
        if isinstance(value, slice):
            return [value]
        if isinstance(value, list) and all(isinstance(sl, slice) for sl in value):
            return value

        raise TypeError(f"Orbital space {space!r} does not resolve to slice objects.")

    def blocks_for_slice(self, sl: slice) -> list[NDArray]:
        """
        Return independent rotation blocks for a contiguous MO-space slice.

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

        idx = slice_indices(sl)
        if (
            self.irrep_indices is None
            or getattr(self.system, "point_group", "C1").upper() == "C1"
            or idx.size == 0
        ):
            return [idx]

        return blocks_by_labels(sl, self.irrep_indices, self.mo_space.nmo)

    def blocks_for_spaces(
        self,
        spaces: OrbitalSpace | Iterable[OrbitalSpace] | None = None,
        relative_to: OrbitalSpace | None = None,
    ) -> list[NDArray]:
        """
        Return independent rotation blocks for named orbital spaces.

        Parameters
        ----------
        spaces : list[str | slice] or str or slice, optional
            Orbital spaces to split into blocks. If omitted, the default spaces
            passed to ``__init__`` are used. Strings name ``mo_space`` attributes.
        relative_to : str or slice, optional
            If provided, subtract the start of this space from every returned
            block. This is useful when full-space blocks are needed in a local
            subspace coordinate system, for example active-space relative indices.

        Returns
        -------
        list[np.ndarray]
            Contiguous-order index arrays, optionally shifted by ``relative_to``.
        """
        if spaces is None:
            if self.spaces is None:
                raise ValueError("Orbital spaces must be provided.")
            spaces = self.spaces
        else:
            spaces = _normalize_spaces(spaces)

        offset = 0
        if relative_to is not None:
            reference_slices = self.slices_for_space(relative_to)
            if len(reference_slices) != 1:
                raise ValueError("relative_to must resolve to exactly one slice.")
            offset = reference_slices[0].start

        blocks = []
        for space in spaces:
            for sl in self.slices_for_space(space):
                for block in self.blocks_for_slice(sl):
                    blocks.append(block - offset)
        return blocks

    def active_blocks(self, relative: bool = True) -> list[NDArray]:
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
        spaces = ["gas"] if hasattr(self.mo_space, "gas") else ["actv"]
        relative_to = "actv" if relative else None
        return self.blocks_for_spaces(spaces, relative_to=relative_to)
