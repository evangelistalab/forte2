"""Reduced density matrices between SparseStates"""

from typing import Annotated

import numpy
from numpy.typing import NDArray

import forte2.lib.sparse_ops


def compute_a_1rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """Compute the alpha 1-RDM between two SparseStates"""

def compute_b_1rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """Compute the beta 1-RDM between two SparseStates"""

def compute_aa_2rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
    """Compute the alpha-alpha 2-RDM between two SparseStates"""

def compute_ab_2rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
    """Compute the alpha-beta 2-RDM between two SparseStates"""

def compute_bb_2rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None))]:
    """Compute the beta-beta 2-RDM between two SparseStates"""

def compute_aaa_3rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
    """Compute the alpha-alpha-alpha 3-RDM between two SparseStates"""

def compute_aab_3rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
    """Compute the alpha-alpha-beta 3-RDM between two SparseStates"""

def compute_abb_3rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
    """Compute the alpha-beta-beta 3-RDM between two SparseStates"""

def compute_bbb_3rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None))]:
    """Compute the beta-beta-beta 3-RDM between two SparseStates"""

def compute_aaaa_4rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None, None, None))]:
    """Compute the alpha-alpha-alpha-alpha 4-RDM between two SparseStates"""

def compute_aaab_4rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None, None, None))]:
    """Compute the alpha-alpha-alpha-beta 4-RDM between two SparseStates"""

def compute_aabb_4rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None, None, None))]:
    """Compute the alpha-alpha-beta-beta 4-RDM between two SparseStates"""

def compute_abbb_4rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None, None, None))]:
    """Compute the alpha-beta-beta-beta 4-RDM between two SparseStates"""

def compute_bbbb_4rdm(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None, None, None, None, None, None, None))]:
    """Compute the beta-beta-beta-beta 4-RDM between two SparseStates"""

def compute_a_1rdm_complex(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None))]:
    """
    Compute the complex alpha 1-RDM between two SparseStates (conjugates the bra)
    """

def compute_aa_2rdm_complex(state_left: forte2.lib.sparse_ops.SparseState, state_right: forte2.lib.sparse_ops.SparseState, norb: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None, None, None))]:
    """
    Compute the complex alpha-alpha 2-RDM between two SparseStates (conjugates the bra)
    """
