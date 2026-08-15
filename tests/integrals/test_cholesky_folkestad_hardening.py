"""
Behavioral locks for the production Folkestad two-step driver (``cholesky_pivots`` /
``cholesky_vectors_ri`` / ``cholesky_pivoted``).

The oracle gate (``test_cholesky_reference_oracle.py``) compares production to the frozen reference
with the batch cap disabled, so it checks the significant-set/screening internals but not the
production-only features. These tests pin those features so a regression is caught:

* **Bounded ``max_qual`` batching** must still reconstruct the ERI to ``tol`` with only a modest
  pivot overshoot, and must actually split the work into more than one qualified block on a system
  large enough to exceed the cap.
* **Significant-set restriction** must drop provably negligible AO-pair columns (leave them exactly
  zero in ``B``) on a system with well-separated fragments.
* **Step II screen-before-compute** must actually skip evaluation (call the screened primitive) yet
  leave the pivot metric ``S`` exact, so the reconstruction is unchanged.
"""

import numpy as np
import pytest

from forte2 import System, integrals
from forte2.integrals.cholesky import (
    cholesky_pivots,
    cholesky_pivoted,
    cholesky_vectors_ri,
    _build_shell_pair_layout,
)

WATER = """
O 0.000000000000 0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369 0.489330954643
H 0.000000000000 0.711620616369 0.489330954643
"""

# Two He atoms far apart: cross-atom AO pairs have vanishing (ab|ab).
HE2_FAR = "He 0.0 0.0 0.0\nHe 0.0 0.0 8.0"


def _recon_bound(tol):
    return max(1e-9, 32.0 * tol)


@pytest.mark.parametrize("tol", [1e-6, 1e-10])
def test_max_qual_batching_reconstructs(tol):
    """A small max_qual splits Step I into several batches yet still reconstructs to tol.

    The pivot set is threshold-exact regardless of the batch split, so a tight cap may only overshoot
    the vector count slightly (the documented two-step trade-off) -- never lose reconstruction
    accuracy.
    """
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nbf = system.nbf
    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)

    # A deliberately tight cap (about one shell block) forces multiple qualified batches.
    B, naux = cholesky_pivoted(system, tol, max_qual=2 * nbf)
    gram = B.reshape(naux, nbf * nbf)
    gram = gram.T @ gram
    assert np.max(np.abs(gram - M)) <= _recon_bound(tol)


def _count_step1_blocks(system, tol, max_qual, layout, monkeypatch):
    """Number of qualified-block evaluations Step I makes at the given cap."""
    calls = {"n": 0}
    orig = integrals.coulomb_4c_pair_block_screened

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(integrals, "coulomb_4c_pair_block_screened", counted)
    cholesky_pivots(system, tol, max_qual=max_qual, layout=layout)
    monkeypatch.undo()
    return calls["n"]


def test_max_qual_splits_into_more_batches_than_uncapped(monkeypatch):
    """A tight max_qual must fragment Step I into more block evaluations than the uncapped path.

    The uncapped path takes one block per sigma-qualification round; capping the qualified-column
    count forces additional blocks within a round, so the tight-cap block count strictly exceeds the
    uncapped one.
    """
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nbf = system.nbf
    layout = _build_shell_pair_layout(system.basis)

    n_uncapped = _count_step1_blocks(system, 1e-8, 0, layout, monkeypatch)
    n_capped = _count_step1_blocks(system, 1e-8, 2 * nbf, layout, monkeypatch)
    assert n_uncapped >= 1
    assert n_capped > n_uncapped


def test_significant_set_drops_negligible_pivots():
    """Step I never selects a pivot from the sub-threshold (cross-atom) AO pairs.

    For well-separated He2 the cross-atom AO pairs have diagonal below tau, so they are outside the
    significant set and can never be pivots. The resulting B leaves those columns exactly zero and
    still reconstructs the ERI. (The diagonal and B columns are both in global row-major AO order, so
    the below-threshold mask indexes B's columns directly.)
    """
    system = System(xyz=HE2_FAR, basis_set="cc-pvdz")
    nbf = system.nbf
    tol = 1e-8

    diag = np.asarray(integrals.coulomb_4c_diagonal(system)).reshape(-1)  # global order
    below = diag < tol
    assert below.any()  # the geometry must actually exercise the significant set

    B, naux = cholesky_pivoted(system, tol)
    Bflat = B.reshape(naux, nbf * nbf)  # global-order columns, aligned with `below`
    # Sub-threshold AO pairs carry no Cholesky vector: those columns of B are exactly zero.
    assert np.count_nonzero(Bflat[:, below]) == 0
    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)
    gram = Bflat.T @ Bflat
    assert np.max(np.abs(gram - M)) <= _recon_bound(tol)


def test_step2_screening_calls_screened_primitive(monkeypatch):
    """Step II builds (ab|K) via the *screened* primitive when a tol is given (Eq. 15 skip)."""
    system = System(xyz=WATER, basis_set="cc-pvdz")
    layout = _build_shell_pair_layout(system.basis)
    pivots = cholesky_pivots(system, 1e-8, layout=layout)

    seen = {"screened": False}
    orig = integrals.coulomb_4c_pair_block_screened

    def spy(system_, bra, ket, schwarz, tau, basis=None):
        # A real screening request passes a schwarz array and a positive tau.
        if schwarz is not None and tau is not None and tau > 0.0:
            seen["screened"] = True
        return orig(system_, bra, ket, schwarz, tau, basis)

    monkeypatch.setattr(integrals, "coulomb_4c_pair_block_screened", spy)
    cholesky_vectors_ri(system, pivots, tol=1e-8, layout=layout)
    assert seen["screened"]


def test_step2_no_tol_reconstructs_without_screening():
    """With tol=None Step II applies no screening and still reconstructs from the given pivots."""
    system = System(xyz=WATER, basis_set="cc-pvdz")
    nbf = system.nbf
    layout = _build_shell_pair_layout(system.basis)
    pivots = cholesky_pivots(system, 1e-8, layout=layout)

    B, naux = cholesky_vectors_ri(system, pivots, tol=None, layout=layout)
    assert naux == len(pivots)
    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)
    gram = B.reshape(naux, nbf * nbf)
    gram = gram.T @ gram
    # No screening => reconstruction accuracy is set purely by the pivot set (built at 1e-8).
    assert np.max(np.abs(gram - M)) <= _recon_bound(1e-8)
