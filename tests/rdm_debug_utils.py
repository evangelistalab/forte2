"""
Generic SparseState-based reference RDM implementations, used only to validate the fast
(block-addressed or hash-map based) production RDM kernels against a slow, independent
implementation. Test-only: production code never imports this module.
"""

from forte2.lib.sparse_ops import SparseState
from forte2.lib import rdms
from forte2.ci.ci_utils import make_2cumulant_so


def sparse_state_from_ci_vector(dets, coefficients):
    """
    Build a `SparseState` from a list of determinants and a coefficient vector.

    Parameters
    ----------
    dets : Sequence[Determinant]
        The determinants spanning the coefficient vector.
    coefficients : np.ndarray
        The CI coefficients, one per determinant.

    Returns
    -------
    SparseState
        A sparse state mapping each determinant to its coefficient.
    """
    return SparseState({d: c for d, c in zip(dets, coefficients)})


def make_so_1rdm_debug(solver, left_root: int, right_root: int | None = None):
    """
    Spin-orbital 1-RDM for two roots of a two-component single-state solver (``_RelCISingleStateSolver``
    or ``_RelSelectedCISingleStateSolver``), via the generic SparseState reference implementation.
    """
    if right_root is None:
        right_root = left_root
    left = sparse_state_from_ci_vector(solver.dets, solver.evecs[:, left_root])
    right = sparse_state_from_ci_vector(solver.dets, solver.evecs[:, right_root])
    return rdms.compute_1rdm_2c(left, right, solver.norb)


def make_so_2rdm_debug(solver, left_root: int, right_root: int | None = None):
    """Spin-orbital 2-RDM reference; see `make_so_1rdm_debug`."""
    if right_root is None:
        right_root = left_root
    left = sparse_state_from_ci_vector(solver.dets, solver.evecs[:, left_root])
    right = sparse_state_from_ci_vector(solver.dets, solver.evecs[:, right_root])
    return rdms.compute_2rdm_2c(left, right, solver.norb)


def make_so_3rdm_debug(solver, left_root: int, right_root: int | None = None):
    """Spin-orbital 3-RDM reference; see `make_so_1rdm_debug`."""
    if right_root is None:
        right_root = left_root
    left = sparse_state_from_ci_vector(solver.dets, solver.evecs[:, left_root])
    right = sparse_state_from_ci_vector(solver.dets, solver.evecs[:, right_root])
    return rdms.compute_3rdm_2c(left, right, solver.norb)


def make_so_2cumulant_debug(solver, left_root: int, right_root: int | None = None):
    """Spin-orbital 2-cumulant reference; see `make_so_1rdm_debug`."""
    rdm1 = make_so_1rdm_debug(solver, left_root, right_root)
    rdm2 = make_so_2rdm_debug(solver, left_root, right_root)
    return make_2cumulant_so(rdm1, rdm2)
