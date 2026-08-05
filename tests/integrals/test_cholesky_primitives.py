"""
Tests for the four-center two-electron integral primitives used by on-the-fly Cholesky
decomposition of the ERIs: the diagonal ``(mn|mn)`` and dense shell-pair blocks ``(AB|CD)``.

Each primitive is validated on both integral backends:
  * against the dense ``coulomb_4c`` tensor (the numerical oracle), and
  * against each other (libint2 vs libcint must agree element-wise).
"""

import numpy as np
import pytest

from forte2 import System, integrals
from forte2.lib import ints
from forte2.integrals import LIBCINT_AVAILABLE


def _diagonal_ref(V):
    """Diagonal (mn|mn) from the dense tensor, row-major over p = m*nbf + n."""
    return np.einsum("mnmn->mn", V).ravel()


def _pair_block_ref(V, basis, bra_pairs, ket_pairs):
    """Reference block (AB|CD) built directly from the dense tensor.

    Within a shell-pair the AO-pair order is iA*nB + iB; blocks are concatenated in the given
    shell-pair order. This mirrors the layout produced by ``coulomb_4c_pair_block``.
    """
    first_size = basis.shell_first_and_size

    def ao_indices(pairs):
        idx = []
        for A, B in pairs:
            fA, nA = first_size[A]
            fB, nB = first_size[B]
            for iA in range(nA):
                for iB in range(nB):
                    idx.append((fA + iA, fB + iB))
        return idx

    rows = ao_indices(bra_pairs)
    cols = ao_indices(ket_pairs)
    out = np.empty((len(rows), len(cols)))
    for r, (m, n) in enumerate(rows):
        for c, (p, q) in enumerate(cols):
            out[r, c] = V[m, n, p, q]
    return out


def _sample_pairs(nshells):
    """A deterministic, diverse set of shell pairs (diagonal, off-diagonal, transposed)."""
    pairs = []
    for a in range(nshells):
        pairs.append((a, a))  # diagonal shell-pairs
    if nshells >= 2:
        pairs += [(0, nshells - 1), (nshells - 1, 0), (0, 1), (1, 0)]
    if nshells >= 3:
        pairs += [(1, 2), (nshells - 1, nshells - 2)]
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# libint2 vs dense coulomb_4c
# ---------------------------------------------------------------------------
def test_coulomb_4c_diagonal_libint2():
    xyz = """
    O 0.000000000000     0.000000000000    -0.061664597388
    H 0.000000000000    -0.711620616370     0.489330954643
    H 0.000000000000     0.711620616370     0.489330954643
    """
    system = System(xyz=xyz, basis_set="sto-3g")
    V = integrals.coulomb_4c(system)  # libint2 dense oracle

    diag = ints.coulomb_4c_diagonal(system.basis)  # libint2 primitive
    assert diag.shape == (system.basis.size**2,)
    assert np.all(diag >= 0.0)  # (mn|mn) is non-negative
    assert np.allclose(diag, _diagonal_ref(V), atol=1e-10, rtol=0)


def test_coulomb_4c_pair_block_libint2():
    xyz = """
    O 0.0 0.0 0.0
    H 0.0 0.0 0.95
    H 0.90 0.0 -0.30
    """
    system = System(xyz=xyz, basis_set="cc-pvdz")  # includes d shells (l=2)
    V = integrals.coulomb_4c(system)
    nshells = system.basis.nshells
    pairs = _sample_pairs(nshells)

    # full block
    block = integrals.coulomb_4c_pair_block(system, pairs, pairs)
    assert np.allclose(
        block, _pair_block_ref(V, system.basis, pairs, pairs), atol=1e-10, rtol=0
    )

    # single ket pair (the Cholesky "pivot column" use-case) and single bra pair
    one = [pairs[-1]]
    col = integrals.coulomb_4c_pair_block(system, pairs, one)
    assert np.allclose(
        col, _pair_block_ref(V, system.basis, pairs, one), atol=1e-10, rtol=0
    )
    row = integrals.coulomb_4c_pair_block(system, one, pairs)
    assert np.allclose(
        row, _pair_block_ref(V, system.basis, one, pairs), atol=1e-10, rtol=0
    )


def test_coulomb_4c_pair_block_matches_diagonal_libint2():
    """The (mn|mn) entries embedded in a block must match the dedicated diagonal primitive."""
    xyz = "Ne 0 0 0"
    system = System(xyz=xyz, basis_set="cc-pvdz")
    diag = ints.coulomb_4c_diagonal(system.basis)
    first_size = system.basis.shell_first_and_size
    nbf = system.basis.size
    nshells = system.basis.nshells
    pairs = [(a, b) for a in range(nshells) for b in range(nshells)]
    block = integrals.coulomb_4c_pair_block(system, pairs, pairs)

    r = 0
    for A, B in pairs:
        fA, nA = first_size[A]
        fB, nB = first_size[B]
        for iA in range(nA):
            for iB in range(nB):
                m, n = fA + iA, fB + iB
                # the (m,n)-(m,n) entry lives on the block diagonal for this pair vs itself
                # find the column index of ket-pair (A,B), same offset
                # (rows and cols use identical pair list/order, so diagonal is at [r, r])
                assert block[r, r] == pytest.approx(diag[m * nbf + n], rel=0, abs=1e-10)
                r += 1


# ---------------------------------------------------------------------------
# libcint vs dense coulomb_4c, and libint2 vs libcint (cross-backend agreement)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_coulomb_4c_diagonal_libcint():
    xyz = """
    O 0.000000000000     0.000000000000    -0.061664597388
    H 0.000000000000    -0.711620616370     0.489330954643
    H 0.000000000000     0.711620616370     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pvdz")
    V = integrals.coulomb_4c(system)

    diag_cint = integrals.cint_coulomb_4c_diagonal(system)  # libcint primitive
    diag_int2 = ints.coulomb_4c_diagonal(system.basis)  # libint2 primitive

    assert np.allclose(diag_cint, _diagonal_ref(V), atol=1e-8, rtol=0)
    # cross-backend agreement
    assert np.linalg.norm(diag_cint - diag_int2) < 1e-8


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_coulomb_4c_pair_block_libcint():
    xyz = """
    O 0.0 0.0 0.0
    H 0.0 0.0 0.95
    H 0.90 0.0 -0.30
    """
    system = System(xyz=xyz, basis_set="cc-pvdz")
    V = integrals.coulomb_4c(system)
    nshells = system.basis.nshells
    pairs = _sample_pairs(nshells)

    block_cint = integrals.cint_coulomb_4c_pair_block(system, pairs, pairs)
    block_int2 = integrals.coulomb_4c_pair_block(system, pairs, pairs)

    assert np.allclose(
        block_cint, _pair_block_ref(V, system.basis, pairs, pairs), atol=1e-8, rtol=0
    )
    # cross-backend agreement (element-wise)
    assert np.linalg.norm(block_cint - block_int2) < 1e-8
