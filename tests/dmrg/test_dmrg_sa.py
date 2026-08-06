"""
DMRG (block2) tight-interface tests: state averaging / multiple roots.

DMRGSolver must reach full state-averaging parity with CISolver: multiple States
and multiple roots per state, with per-state sub-solvers. Reference energies are
computed in-test with the exact CI solver on the same active space.
"""

import numpy as np

from forte2 import System, RHF, State, CI
from forte2.dmrg import DMRG
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2

TIGHT_DMRG_PARAMS = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=12,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)

N2_KWARGS = dict(
    core_orbitals=[0, 1, 2, 3],
    active_orbitals=[4, 5, 6, 7, 8, 9],
)


def _n2_system():
    return System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.2",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )


@requires_block2
def test_dmrg_multiple_roots_same_state(tmp_path):
    """Two roots of the same singlet state, state-averaged, vs FCI."""
    system = _n2_system()
    state = State(nel=14, multiplicity=1, ms=0.0)

    hf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(states=state, nroots=2, **N2_KWARGS)(hf)
    ci.run()

    hf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=state,
        nroots=2,
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
        **N2_KWARGS,
    )(hf)
    dmrg.run()

    assert len(dmrg.E) == 2
    assert dmrg.E[0] == approx(ci.E[0])
    assert dmrg.E[1] == approx(ci.E[1])
    # equal weights by default
    assert dmrg.compute_average_energy() == approx(0.5 * (ci.E[0] + ci.E[1]))


@requires_block2
def test_dmrg_multiple_states(tmp_path):
    """Singlet + triplet, multiple roots each, state-averaged, vs FCI."""
    system = _n2_system()
    singlet = State(nel=14, multiplicity=1, ms=0.0)
    triplet = State(nel=14, multiplicity=3, ms=0.0)

    weights = [[1.0], [0.85, 0.15]]
    hf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        states=[singlet, triplet],
        nroots=[1, 2],
        weights=weights,
        **N2_KWARGS,
    )(hf)
    ci.run()

    hf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=[singlet, triplet],
        nroots=[1, 2],
        weights=weights,
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
        **N2_KWARGS,
    )(hf)
    dmrg.run()

    assert len(dmrg.sub_solvers) == 2
    assert len(dmrg.E) == 3
    for i in range(3):
        assert dmrg.E[i] == approx(ci.E[i])
    assert dmrg.compute_average_energy() == approx(ci.compute_average_energy())


@requires_block2
def test_dmrg_sa_evals_per_solver_layout(tmp_path):
    """evals_per_solver mirrors the per-state root structure like CISolver."""
    system = _n2_system()
    singlet = State(nel=14, multiplicity=1, ms=0.0)
    triplet = State(nel=14, multiplicity=3, ms=0.0)

    hf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=[singlet, triplet],
        nroots=[1, 2],
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
        **N2_KWARGS,
    )(hf)
    dmrg.run()

    assert len(dmrg.evals_per_solver) == 2
    assert len(dmrg.evals_per_solver[0]) == 1
    assert len(dmrg.evals_per_solver[1]) == 2
    assert np.allclose(np.concatenate(dmrg.evals_per_solver), dmrg.E)
