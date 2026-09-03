from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from forte2.helpers import logger

from .iao import IBO
from .ibo_align import IBOAligner
from .natural_orbitals import NaturalOrbitals
from .orbital_blocks import OrbitalBlockBuilder
from .semicanonicalizer import Semicanonicalizer

FinalOrbitals = Literal["original", "semicanonical", "natural", "ibo", "ibo_atomic"]

VALID_FINAL_ORBITALS = get_args(FinalOrbitals)


def make_final_orbitals(
    mode: str,
    *,  # the following args are keyword-only
    system,
    mo_space,
    irrep_indices: ArrayLike,
    C_contig: NDArray,
    g1_act: NDArray | None,
) -> NDArray:
    """
    Build the requested final orbitals, in contiguous MO-space ordering.

    For every mode except ``"original"``, the inactive subspaces are
    semicanonicalized. For ``mode``:

    - ``"semicanonical"``: The active space is semicanonicalized.
    - ``"natural"``: The active space is rotated to make the active 1-RDM diagonal.
    - ``"ibo"``: The active orbitals are localized as intrinsic bond orbitals (IBOs).
    - ``"ibo_atomic"``: The active orbitals are localized and aligned to the
      global axis-oriented IAOs and ordered by atom and native MINAO
      basis-function index. Localization and ordering are applied separately
      within each GAS partition, and both modes are available only in C1
      symmetry.

    Parameters
    ----------
    mode : str
        Either "original", "semicanonical", "natural", "ibo", or
        "ibo_atomic".
    system : System
        The system, used to build the generalized Fock matrix.
    mo_space : MOSpace or EmbeddingMOSpace
        The MO-space partition.
    irrep_indices : ArrayLike
        Orbital irrep labels in the same contiguous ordering as C_contig.
    C_contig : NDArray
        Full MO coefficient matrix in contiguous MO-space ordering.
    g1_act : NDArray or None
        Active-space one-particle density matrix (spin-summed if non-relativistic,
        spin-orbital if relativistic). May be ``None`` only for ``"original"``.

    Returns
    -------
    NDArray
        The transformed coefficient matrix, still in contiguous ordering.
    """
    if mode not in VALID_FINAL_ORBITALS:
        raise ValueError(
            f"final_orbitals must be one of {VALID_FINAL_ORBITALS}, "
            f"but got {mode!r}."
        )

    C_final = np.asarray(C_contig).copy()
    if mode == "original":
        return C_final

    ibo_mode = mode in ("ibo", "ibo_atomic")
    if ibo_mode:
        point_group = str(getattr(system, "point_group", "C1")).upper()
        if point_group != "C1":
            raise ValueError(
                f"final_orbitals={mode!r} is only available in C1 symmetry, but the "
                f"system uses point group {point_group!r}. Construct the System with "
                "symmetry=False to disable point-group symmetry."
            )
        if getattr(system, "two_component", False) or np.iscomplexobj(C_contig):
            raise NotImplementedError(
                f"final_orbitals={mode!r} is currently implemented only for real, "
                "nonrelativistic orbitals."
            )

    # Semicanonicalize every inactive subspace. The active space is included
    # only for semicanonical mode; natural and IBO modes define it separately.
    semi = Semicanonicalizer(
        mo_space=mo_space,
        system=system,
        irrep_indices=irrep_indices,
        mix_inactive=False,
        mix_active=False,
        do_active=(mode == "semicanonical"),
    )
    semi.semi_canonicalize(g1=g1_act, C_contig=C_final)
    C_final = semi.C_semican.copy()

    if ibo_mode:
        # Rotations across GAS partitions change the variational space of a GAS
        # calculation, so localize each active partition independently. Since
        # do_active=False above, these columns are still the input active MOs.
        orbital_blocks = OrbitalBlockBuilder(mo_space)
        for active_block in orbital_blocks.active_blocks(relative_index=False):
            if active_block.size < 2 and mode == "ibo":
                continue
            ibo = IBO(system, C_final[:, active_block])
            if mode == "ibo_atomic":
                ibo_aligner = IBOAligner(ibo)
                ibo_aligner.align_to_atomic_orbitals()
                C_final[:, active_block] = ibo_aligner.C_ibo
            else:
                C_final[:, active_block] = ibo.C_ibo
        return C_final

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
