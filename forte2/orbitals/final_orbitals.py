from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from forte2.helpers import logger

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


def check_final_orbital_energy_invariance(
    *,  # keyword-only
    hard_fail: bool,
    tol: float,
    old_E: ArrayLike,
    new_E: ArrayLike,
    old_E_avg: float,
    new_E_avg: float,
    hard_fail_hint: str,
) -> None:
    """
    Compare per-root and average energies before/after a ``final_orbitals``
    rotation, as a sanity check that the change of basis didn't accidentally land
    the solver on a different converged solution.

    Parameters
    ----------
    hard_fail : bool
        Whether the caller expects exact invariance (e.g. full CI, where any
        deviation past the tolerance means the re-solve converged onto a different
        root). If True, a violation raises ``RuntimeError``. If False (e.g. selected
        CI, a variational truncation that is not exactly invariant to orbital
        rotations by construction), a violation is only logged.
    tol : float
        Base tolerance. ``hard_fail=True`` compares against ``10 * tol``, to allow
        for near-threshold numerical noise; ``hard_fail=False`` compares against
        ``tol`` directly.
    old_E, new_E : ArrayLike
        Per-root energies before and after the rotation.
    old_E_avg, new_E_avg : float
        State-averaged energy before and after the rotation.
    hard_fail_hint : str
        Extra suggestion appended to the warning when ``hard_fail`` is True (e.g.
        "Consider increasing davidson_liu_params.maxiter.").

    Raises
    ------
    RuntimeError
        If ``hard_fail`` is True and the energies differ by more than ``10 * tol``.
    """
    max_root_de = np.max(np.abs(np.asarray(old_E) - np.asarray(new_E)))
    avg_de = np.abs(old_E_avg - new_E_avg)
    max_de = max(max_root_de, avg_de)

    threshold = tol * 10.0 if hard_fail else tol
    if max_de <= threshold:
        logger.log_info1(
            f"\nAfter producing the final orbitals, the CI solver converged to\n"
            f"within threshold {threshold}: max(abs(E_i - E_new_i)) = {max_root_de:.4e},\n"
            f"abs(E_avg - E_avg_new) = {avg_de:.4e}.\n"
        )
        return

    if hard_fail:
        logger.log_warning(
            f"After producing the final orbitals, the CI solver converged to "
            f"different solutions: max(abs(E_i - E_new_i)) = {max_root_de:.4e}, "
            f"abs(E_avg - E_avg_new) = {avg_de:.4e}."
        )
        logger.log_warning(hard_fail_hint)
        raise RuntimeError(
            "After producing the final orbitals, the CI solver converged to "
            "different roots."
        )
    else:
        logger.log_warning(
            "The active-space solver is not invariant to final orbital "
            f"rotations; the final-basis energies changed by up to {max_de:.4e}."
        )
