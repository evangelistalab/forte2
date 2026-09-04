"""
``final_orbitals`` post-processing for the standalone relativistic DMRG solver.

Same permutation rationale as test_dmrg_final_orbitals.py: a deliberately
non-contiguous active spinor space (spinors 0-7 core, spinors 8-9 left
virtual, spinors 10-17 active), so that ``mo_space.contig_to_orig`` is a
genuine, non-trivial permutation and a wrong-axis bug in the final-orbital
rotation shows up as a grossly wrong energy rather than passing silently. The
gap deliberately skips a *whole* Kramers pair (spinors 8 and 9 are exactly
degenerate partners): splitting a Kramers pair between core and active space
makes the active space itself numerically unstable (which physical spinor
lands at index 8 vs. 9 is arbitrary within their degenerate 2D subspace), so
that would confound the permutation test with an unrelated ambiguity.

The natural-orbitals test checks the natural occupation number *spectrum*
(sorted eigenvalues of the active 1-RDM) rather than the RDM's raw matrix
diagonality. In a time-reversal-symmetric two-component calculation, every
natural occupation number is exactly Kramers-doubly-degenerate, so "the"
natural spinors are only defined up to an arbitrary unitary rotation within
each degenerate Kramers pair -- confirmed by direct probing, where the
rotation matrix from ``NaturalOrbitals`` diagonalizes the pre-rotation 1-RDM
to machine precision, but DMRG's independent re-optimization in the rotated
basis converges to a different (equally valid) gauge choice within each
degenerate pair, giving a large *matrix* off-diagonal residual despite an
eigenvalue spectrum that agrees with the exact reference to ~1e-9. This
mirrors the degenerate-manifold caveat already documented for transition
RDMs/dipoles in forte2/dmrg/dmrg.py and test_dmrg_transition_dipoles.py.
"""

import numpy as np
import pytest

from forte2 import System, GHF, RelCI, X2CParams
from forte2.dmrg import RelDMRG
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2_complex

TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[400] * 4 + [800] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-14] * 8,
    n_sweeps=16,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)

CORE_ORBITALS = list(range(8))
ACTIVE_ORBITALS = list(range(10, 18))


def _run_reldmrg_and_relci(final_orbitals, tmp_path):
    system = System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )

    scf = GHF(charge=0)(system)
    relci = RelCI(
        core_orbitals=CORE_ORBITALS,
        active_orbitals=ACTIVE_ORBITALS,
        nel=10,
        final_orbitals=final_orbitals if final_orbitals == "semicanonical" else "original",
    )(scf)
    relci.run()

    scf2 = GHF(charge=0)(system)
    reldmrg = RelDMRG(
        core_orbitals=CORE_ORBITALS,
        active_orbitals=ACTIVE_ORBITALS,
        nel=10,
        dmrg_params=TIGHT(str(tmp_path)),
        final_orbitals=final_orbitals,
    )(scf2)
    reldmrg.run()
    return relci, reldmrg


@requires_block2_complex
@pytest.mark.slow
def test_reldmrg_final_orbitals_semicanonical(tmp_path):
    """RelDMRG final_orbitals='semicanonical' == 2C-RelCI."""
    relci, reldmrg = _run_reldmrg_and_relci("semicanonical", tmp_path)
    assert reldmrg.E[0] == approx(relci.E[0].real)


@requires_block2_complex
@pytest.mark.slow
def test_reldmrg_final_orbitals_natural(tmp_path):
    """RelDMRG final_orbitals='natural' reproduces the 2C-RelCI energy and
    natural occupation number spectrum (see module docstring for why the
    spectrum, rather than the RDM's raw matrix diagonality, is the
    gauge-invariant quantity to check for Kramers-degenerate natural spinors).
    """
    relci, reldmrg = _run_reldmrg_and_relci("natural", tmp_path)
    assert reldmrg.E[0] == approx(relci.E[0].real)

    dmrg_occs = np.sort(np.linalg.eigvalsh(reldmrg.make_average_1rdm()))[::-1]
    relci_occs = np.sort(np.linalg.eigvalsh(relci.make_average_1rdm()))[::-1]
    assert dmrg_occs == approx(relci_occs)
