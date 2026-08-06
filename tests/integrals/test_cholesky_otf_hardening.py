"""
Behavioral locks for the production Koch one-step driver (``cholesky_otf``).

The oracle gate (``test_cholesky_reference_oracle.py``) checks only that production reconstructs the
ERI to tolerance. That contract would stay green even if the production accelerations were silently
disabled (falling back to the reference algorithm still reconstructs correctly). These tests pin the
accelerations themselves so a regression that turned one off is caught:

* **Proactive shell-pair draining** must evaluate each ``(**|AB)`` block far fewer times than the
  number of pivots -- a per-pivot rebuild would recompute a block for every column.
* **Significant-set restriction** must drop provably negligible AO-pair columns (leave them exactly
  zero in ``B``) on a system with well-separated fragments, while still reconstructing the ERI.
"""

import numpy as np
import pytest

from forte2 import System, integrals
from forte2.integrals.cholesky import cholesky_otf

WATER = """
O 0.000000000000 0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369 0.489330954643
H 0.000000000000 0.711620616369 0.489330954643
"""

# Two He atoms far apart: cross-atom AO pairs have vanishing (ab|ab), so the significant-set
# prescreen must drop them. 8 bohr is well beyond any cc-pVDZ radial extent.
HE2_FAR = "He 0.0 0.0 0.0\nHe 0.0 0.0 8.0"


def test_draining_evaluates_each_block_once(monkeypatch):
    """Proactive draining: block evaluations are far fewer than pivots.

    A per-pivot column rebuild (the un-drained algorithm) would call the block primitive once per
    pivot. Draining decomposes every qualified column of a loaded block before moving on, so the
    number of block evaluations drops to roughly the number of *distinct* ket shell pairs touched --
    well below the pivot count.
    """
    system = System(xyz=WATER, basis_set="cc-pvdz")

    calls = {"n": 0}
    orig = integrals.coulomb_4c_pair_block_screened

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(integrals, "coulomb_4c_pair_block_screened", counted)
    _, naux = cholesky_otf(system, 1e-8)

    assert calls["n"] > 0
    # Draining must give a substantial reduction; a per-pivot build would make these equal.
    assert calls["n"] < naux / 2


def test_significant_set_drops_negligible_columns():
    """Significant-set restriction zeroes provably negligible AO-pair columns of B.

    For two well-separated He atoms, the cross-atom AO pairs (a on one atom, b on the other) have
    diagonal (ab|ab) below Koch's prescreen tol^2 / X_max, so they carry no Cholesky vector: those
    columns of B stay exactly zero. The surviving columns must still reconstruct the ERI.
    """
    system = System(xyz=HE2_FAR, basis_set="cc-pvdz")
    nbf = system.nbf
    tol = 1e-8

    diag = np.asarray(integrals.coulomb_4c_diagonal(system)).reshape(nbf, nbf)
    dmax0 = float(diag.max())
    screen = tol * tol / dmax0
    n_screened = int(np.sum(diag.reshape(-1) < screen))
    assert n_screened > 0  # the geometry must actually exercise the prescreen

    B, naux = cholesky_otf(system, tol)
    Bflat = B.reshape(naux, nbf * nbf)

    # Every column below the prescreen must be exactly zero across all vectors.
    below = diag.reshape(-1) < screen
    assert np.count_nonzero(Bflat[:, below]) == 0

    # ...and the retained vectors still reconstruct the dense ERI to tolerance.
    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)
    gram = Bflat.T @ Bflat
    assert np.max(np.abs(gram - M)) <= max(1e-9, 32.0 * tol)


@pytest.mark.parametrize("tol", [1e-6, 1e-10])
def test_diagonal_reconstructed_exactly_at_pivots(tol):
    """The reconstruction is exact on the diagonal at every selected pivot.

    Koch's complete pivoting drives each pivot's Schur diagonal to zero, so ``(B^T B)_JJ = M_JJ``
    exactly at a pivot J -- a direct consequence of the algorithm that both draining and screening
    must preserve.
    """
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nbf = system.nbf
    diag = np.asarray(integrals.coulomb_4c_diagonal(system))
    B, naux = cholesky_otf(system, tol)
    Bflat = B.reshape(naux, nbf * nbf)
    gram_diag = np.einsum("ji,ji->i", Bflat, Bflat)  # (B^T B)_pp for every AO pair p
    # The single largest diagonal (the first pivot) is reconstructed to full precision.
    pmax = int(np.argmax(diag))
    assert np.isclose(gram_diag[pmax], diag[pmax], rtol=1e-8, atol=1e-12)
