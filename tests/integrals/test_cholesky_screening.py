"""
Tests for the Cauchy-Schwarz screening primitives used to accelerate on-the-fly Cholesky
decomposition of the ERIs: the shell-pair Schwarz factors ``Q_AB = sqrt(max_{a,b} (ab|ab))`` and the
Schwarz-screened four-center shell-pair block.

The primitives are validated against the dense ``coulomb_4c`` tensor (the numerical oracle):
  * ``Q_AB`` must equal the exact shell-pair max of the diagonal,
  * the screened block must equal the unscreened block exactly when ``tau`` is below the true
    Cauchy-Schwarz bound (screening then removes nothing), and to within ``tau`` for larger ``tau``,
  * screening must actually skip work (produce genuine zeros) at a loose threshold.
"""

import numpy as np
import pytest

from forte2 import System, integrals
from forte2.lib import ints

WATER = """
O 0.0 0.0 0.0
H 0.0 0.0 0.95
H 0.90 0.0 -0.30
"""


def _schwarz_ref(system):
    """Reference Q_AB = sqrt(max_{a in A, b in B} (ab|ab)) from the dense tensor."""
    basis = system.basis
    nbf = basis.size
    V = integrals.coulomb_4c(system)
    diag = np.einsum("mnmn->mn", V)  # (nbf, nbf), = (ab|ab)
    first_size = basis.shell_first_and_size
    nsh = basis.nshells
    Q = np.zeros(nsh * nsh)
    for A in range(nsh):
        fA, nA = first_size[A]
        for B in range(nsh):
            fB, nB = first_size[B]
            block = diag[fA : fA + nA, fB : fB + nB]
            Q[A * nsh + B] = np.sqrt(block.max())
    return Q


def _all_pairs(nsh):
    return np.array([(a, b) for a in range(nsh) for b in range(nsh)], dtype=np.int32)


def test_schwarz_factors_match_dense():
    system = System(xyz=WATER, basis_set="cc-pvdz")  # includes d shells
    Q = np.asarray(ints.coulomb_4c_schwarz_factors(system.basis))
    nsh = system.basis.nshells
    assert Q.shape == (nsh * nsh,)
    assert np.all(Q >= 0.0)
    assert np.allclose(Q, _schwarz_ref(system), atol=1e-10, rtol=0)


def test_schwarz_factors_symmetric():
    """Q_AB == Q_BA since (ab|ab) is symmetric under swapping the pair."""
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nsh = system.basis.nshells
    Q = np.asarray(ints.coulomb_4c_schwarz_factors(system.basis)).reshape(nsh, nsh)
    assert np.allclose(Q, Q.T, atol=1e-12, rtol=0)


def test_screened_block_exact_below_bound():
    """With tau below the true CS bound, screening removes nothing: exact agreement."""
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nsh = system.basis.nshells
    pairs = _all_pairs(nsh)
    Q = np.ascontiguousarray(ints.coulomb_4c_schwarz_factors(system.basis))

    unscreened = np.asarray(integrals.coulomb_4c_pair_block(system, pairs, pairs))
    # A tiny threshold cannot exceed any nonzero Q_AB*Q_CD product, so nothing is dropped.
    screened = np.asarray(
        ints.coulomb_4c_pair_block_screened(system.basis, pairs, pairs, Q, 1e-14)
    )
    assert np.array_equal(screened, unscreened)


@pytest.mark.parametrize("tau", [1e-10, 1e-6, 1e-3])
def test_screened_block_within_tau(tau):
    """The screened block differs from the exact block by at most tau elementwise."""
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nsh = system.basis.nshells
    pairs = _all_pairs(nsh)
    Q = np.ascontiguousarray(ints.coulomb_4c_schwarz_factors(system.basis))

    unscreened = np.asarray(integrals.coulomb_4c_pair_block(system, pairs, pairs))
    screened = np.asarray(
        ints.coulomb_4c_pair_block_screened(system.basis, pairs, pairs, Q, tau)
    )
    # Every dropped entry was provably below tau by Cauchy-Schwarz, so the error is bounded by tau.
    assert np.max(np.abs(screened - unscreened)) <= tau


def test_screening_actually_skips_at_loose_threshold():
    """A loose threshold must zero at least some entries that the exact block leaves nonzero."""
    # Two well-separated atoms give small cross Q_AB, so a moderate tau drops the cross quartets.
    system = System(xyz="He 0.0 0.0 0.0\nHe 0.0 0.0 6.0", basis_set="cc-pvdz")
    nsh = system.basis.nshells
    pairs = _all_pairs(nsh)
    Q = np.ascontiguousarray(ints.coulomb_4c_schwarz_factors(system.basis))

    unscreened = np.asarray(integrals.coulomb_4c_pair_block(system, pairs, pairs))
    screened = np.asarray(
        ints.coulomb_4c_pair_block_screened(system.basis, pairs, pairs, Q, 1e-3)
    )
    dropped = (screened == 0.0) & (unscreened != 0.0)
    assert dropped.any()
    # Everything dropped was below the threshold.
    assert np.all(np.abs(unscreened[dropped]) <= 1e-3)


def test_python_wrapper_none_schwarz_is_exact():
    """The Python wrapper falls back to the exact unscreened block when schwarz is None."""
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nsh = system.basis.nshells
    pairs = _all_pairs(nsh)
    exact = np.asarray(integrals.coulomb_4c_pair_block(system, pairs, pairs))
    fallback = np.asarray(
        integrals.coulomb_4c_pair_block_screened(system, pairs, pairs, None, 1e-8)
    )
    assert np.array_equal(fallback, exact)
