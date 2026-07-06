from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .orbital_blocks import OrbitalBlockBuilder


def _validate_natural_orbital_blocks(
    blocks: Iterable[ArrayLike],
    nactv: int,
    require_complete: bool = False,
) -> list[NDArray]:
    normalized_blocks = []
    seen = np.zeros(nactv, dtype=bool)

    for block in blocks:
        idx = np.asarray(block, dtype=int)
        if idx.size == 0:
            normalized_blocks.append(idx)
            continue
        if idx.ndim != 1:
            raise ValueError("Each natural-orbital block must be one-dimensional.")
        if np.any(idx < 0) or np.any(idx >= nactv):
            raise ValueError("Natural-orbital block indices are out of bounds.")
        if np.any(seen[idx]):
            raise ValueError("Natural-orbital blocks must not overlap.")
        seen[idx] = True
        normalized_blocks.append(idx)

    if require_complete and not np.all(seen):
        missing = np.flatnonzero(~seen)
        raise ValueError(
            "Natural-orbital blocks must cover the full active space. "
            f"Missing active indices: {missing.tolist()}."
        )

    return normalized_blocks


def make_natural_orbitals(
    C_act: ArrayLike,
    g1_act: ArrayLike,
    blocks: Iterable[ArrayLike] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    A function to build active-space natural orbitals from an active-space 1-RDM.

    The transformation is restricted to the supplied blocks. This lets callers
    preserve GAS partitions, point-group irreps, or both, by passing active-space
    relative index blocks.

    This function is intended to be used by the NaturalOrbital class, which handles
    the bookkeeping of the full MO coefficient matrix and the orbital rotation matrix.

    Parameters
    ----------
    C_act : NDArray
        Active molecular orbital coefficients with shape ``(nbasis, nactv)``.
    g1_act : NDArray
        Active-space one-particle density matrix in the basis of ``C_act``.
    blocks : list[NDArray], optional
        Active-space relative orbital index blocks. If omitted, the full active
        space is diagonalized as one block.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        ``C_nat``, ``U_nat``, and ``occupations``. ``U_nat`` rotates the input
        active orbitals to natural orbitals, and ``occupations`` stores natural
        occupation numbers in the returned active orbital order.
    """
    C_act = np.asarray(C_act)
    g1_act = np.asarray(g1_act)
    if C_act.ndim != 2:
        raise ValueError("C_act must be a two-dimensional array.")

    nactv = C_act.shape[1]
    if g1_act.shape != (nactv, nactv):
        raise ValueError("g1_act must have shape (nactv, nactv).")

    if blocks is None:
        blocks = [np.arange(nactv)]
    blocks = _validate_natural_orbital_blocks(blocks, nactv)

    U_nat = np.eye(nactv, dtype=np.result_type(C_act, g1_act))
    occ_dtype = np.result_type(np.real(g1_act).dtype, float)
    occupations = np.asarray(np.real_if_close(np.diag(g1_act)), dtype=occ_dtype).copy()

    for idx in blocks:
        if idx.size == 0:
            continue

        occ, c = np.linalg.eigh(g1_act[np.ix_(idx, idx)])
        order = np.argsort(occ)[::-1]
        occ = occ[order]
        c = c[:, order]

        U_nat[np.ix_(idx, idx)] = c
        occupations[idx] = np.real_if_close(occ)

    return C_act @ U_nat, U_nat, occupations


class NaturalOrbital:
    """
    A helper class to build active-space natural orbitals while preserving
    GAS and symmetry blocks using information from the System and MO space objects.

    Parameters
    ----------
    system: forte2.System
        System object used to determine whether point-group symmetry is active.
    mo_space: forte2.MOSpace | forte2.EmbeddingMOSpace
        MO-space partition. The active-space slice and GAS slices are taken from
        this object.
    irrep_indices : np.ndarray or list[int], optional
        Orbital irrep labels in the same contiguous ordering as ``C_contig``.
        If provided for a non-C1 system, natural orbitals are formed separately
        inside each active-space irrep block and GAS partition.

    Attributes
    ----------
    C_natural : np.ndarray
        Full MO coefficient matrix after replacing active orbitals by natural
        orbitals.
    U : np.ndarray
        Full-space orbital rotation from the input MOs to ``C_natural``.
    Uactv : np.ndarray
        Active-space natural orbital rotation.
    nat_occs : np.ndarray
        Natural occupation numbers in the returned active orbital order.
    """

    def __init__(
        self, system, mo_space, irrep_indices: ArrayLike | None = None
    ) -> None:
        self.system = system
        self.mo_space = mo_space
        self.orbital_blocks = OrbitalBlockBuilder(system, mo_space, irrep_indices)

    def make_natural_orbitals(self, g1_act: ArrayLike, C_contig: ArrayLike) -> None:
        """
        Construct natural orbitals from a full contiguous MO coefficient matrix.

        Parameters
        ----------
        g1_act : np.ndarray
            Active-space one-particle density matrix.
        C_contig : np.ndarray
            Full MO coefficient matrix in contiguous MO-space ordering.
        """
        C_contig = np.asarray(C_contig)
        if C_contig.ndim != 2:
            raise ValueError("C_contig must be a two-dimensional array.")
        if C_contig.shape[1] != self.mo_space.nmo:
            raise ValueError("C_contig must have one column per MO.")

        nactv = self.mo_space.actv.stop - self.mo_space.actv.start
        active_blocks = self.orbital_blocks.active_blocks(relative=True)
        active_blocks = _validate_natural_orbital_blocks(
            active_blocks, nactv, require_complete=True
        )
        C_act_nat, U_act, nat_occs = make_natural_orbitals(
            C_contig[:, self.mo_space.actv],
            g1_act,
            blocks=active_blocks,
        )

        self.C_natural = C_contig.copy()
        self.C_natural[:, self.mo_space.actv] = C_act_nat
        self.U = np.eye(self.mo_space.nmo, dtype=U_act.dtype)
        actv_idx = np.arange(self.mo_space.actv.start, self.mo_space.actv.stop)
        self.U[np.ix_(actv_idx, actv_idx)] = U_act
        self.Uactv = U_act
        self.nat_occs = nat_occs
