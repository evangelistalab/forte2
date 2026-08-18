from typing import Literal, get_args

from numpy.typing import ArrayLike, NDArray

from .natural_orbitals import NaturalOrbitals
from .semicanonicalizer import Semicanonicalizer

FinalOrbitals = Literal["original", "semicanonical", "natural"]

VALID_FINAL_ORBITALS = get_args(FinalOrbitals)


def validate_final_orbitals(value: str) -> None:
    """
    Validate a final_orbitals option value.

    Parameters
    ----------
    value : str
        The requested value.

    Raises
    ------
    ValueError
        If value is not in the FinalOrbitals literal.
    """
    if value not in VALID_FINAL_ORBITALS:
        raise ValueError(
            f"final_orbitals must be one of {VALID_FINAL_ORBITALS}, "
            f"but got {value!r}."
        )
    return


def make_final_orbitals(
    mode: str,
    *,  # the following args are keyword-only
    system,
    mo_space,
    irrep_indices: ArrayLike,
    C_contig: NDArray,
    g1_act: NDArray,
) -> NDArray:
    """
    Build the requested final orbitals, in contiguous MO-space ordering.

    The inactive subspaces are always semicanonicalized. The active space is
    semicanonicalized for mode="semicanonical", or left alone by the
    semicanonicalizer and then diagonalized against the active 1-RDM for
    mode="natural" (separately within each GAS partition and irrep block, so
    those structures are preserved).

    Parameters
    ----------
    mode : str
        Either "semicanonical" or "natural". Callers are expected to skip
        this function entirely for "original".
    system : System
        The system, used to build the generalized Fock matrix.
    mo_space : MOSpace or EmbeddingMOSpace
        The MO-space partition.
    irrep_indices : ArrayLike
        Orbital irrep labels in the same contiguous ordering as C_contig.
    C_contig : NDArray
        Full MO coefficient matrix in contiguous MO-space ordering.
    g1_act : NDArray
        Active-space one-particle density matrix (spin-summed if non-relativistic,
        spin-orbital if relativistic).

    Returns
    -------
    NDArray
        The transformed coefficient matrix, still in contiguous ordering.
    """
    if mode not in ("semicanonical", "natural"):
        raise ValueError(
            "make_final_orbitals expects mode to be 'semicanonical' or 'natural', "
            f"but got {mode}."
        )

    # Semicanonicalize the orbital subspaces (except the CAS/GAS in the case of
    # natural orbitals, which are defined by the 1-RDM instead).
    semi = Semicanonicalizer(
        mo_space=mo_space,
        system=system,
        irrep_indices=irrep_indices,
        mix_inactive=False,
        mix_active=False,
        do_active=(mode == "semicanonical"),
    )
    semi.semi_canonicalize(g1=g1_act, C_contig=C_contig)
    C_final = semi.C_semican.copy()

    if mode == "natural":
        natural_orbital = NaturalOrbitals(mo_space, irrep_indices=irrep_indices)
        natural_orbital.make_natural_orbitals(g1_act=g1_act, C_contig=C_final)
        C_final = natural_orbital.C_natural.copy()

    return C_final
