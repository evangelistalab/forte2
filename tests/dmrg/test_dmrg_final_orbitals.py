"""
``final_orbitals`` post-processing for the standalone DMRG solver.

These tests use a deliberately *non-contiguous* active space (core orbitals
0-3, orbital 4 left virtual, active orbitals 5-10) rather than the
core-then-active-then-virtual layout used elsewhere in the DMRG test suite.
This matters because ``DMRG``/``RelDMRG`` build the semicanonical/natural
orbitals in the MOSpace's *contiguous* ordering and then have to permute the
result back into the original orbital ordering via
``mo_space.contig_to_orig``; with a contiguous-by-construction active space
that permutation happens to be the identity, so a bug in the permutation
(indexing the wrong array axis) would silently pass unnoticed. A
non-contiguous active space makes ``contig_to_orig`` a genuine, non-trivial
permutation, so a wrong axis produces a grossly wrong (not just slightly
noisy) energy after the final-orbital rotation and re-diagononalization.
"""

import numpy as np
import pytest

from forte2 import System, RHF, State, CI
from forte2.dmrg import DMRG
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2

TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=12,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)

XYZ = "N 0.0 0.0 0.0\nN 0.0 0.0 2.0"
CORE_ORBITALS = [0, 1, 2, 3]
ACTIVE_ORBITALS = [5, 6, 7, 8, 9, 10]


def _run_dmrg_and_ci(final_orbitals, tmp_path):
    system = System(
        xyz=XYZ, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    state = State(nel=14, multiplicity=1, ms=0.0)

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=state,
        core_orbitals=CORE_ORBITALS,
        active_orbitals=ACTIVE_ORBITALS,
        final_orbitals=final_orbitals if final_orbitals == "semicanonical" else "original",
    )(rhf)
    ci.run()

    rhf2 = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=state,
        core_orbitals=CORE_ORBITALS,
        active_orbitals=ACTIVE_ORBITALS,
        dmrg_params=TIGHT(str(tmp_path)),
        final_orbitals=final_orbitals,
    )(rhf2)
    dmrg.run()
    return ci, dmrg


@requires_block2
@pytest.mark.slow
def test_dmrg_final_orbitals_semicanonical(tmp_path):
    """DMRG final_orbitals='semicanonical' == FCI (energy is orbital-rotation
    invariant, but a botched final-orbital permutation would break it)."""
    ci, dmrg = _run_dmrg_and_ci("semicanonical", tmp_path)
    assert dmrg.E[0] == approx(ci.E[0])


@requires_block2
@pytest.mark.slow
def test_dmrg_final_orbitals_natural(tmp_path):
    """DMRG final_orbitals='natural' reproduces the FCI energy and yields
    genuine natural orbitals (diagonal active-space 1-RDM)."""
    ci, dmrg = _run_dmrg_and_ci("natural", tmp_path)
    assert dmrg.E[0] == approx(ci.E[0])

    g1_act = dmrg.make_average_1rdm()
    off_diag = g1_act - np.diag(np.diag(g1_act))
    assert np.max(np.abs(off_diag)) < 1e-6
