"""
Relativistic (complex) DMRG tight-interface tests: state averaging.

RelDMRGSolver must reach state-averaging parity with RelCISolver: multiple 2C
roots of a given electron number, with per-state sub-solvers. Reference energies
are computed in-test with the exact 2C-FCI (RelCI) solver on the same space.
"""

import numpy as np

from forte2 import System, GHF, CI, RelCISolver
from forte2.dmrg import RelDMRGSolver
from forte2.base_classes.params import DMRGParams, X2CParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2_complex

TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[250] * 4 + [500] * 6,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 10,
    n_sweeps=16,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


def _na_system():
    # Na atom with spin-orbit X2C: near-degenerate low-lying 2C roots, a good
    # state-averaging stress test that is still small.
    return System(
        xyz="Na 0.0 0.0 0.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )


@requires_block2_complex
def test_reldmrg_multiple_roots(tmp_path):
    """State-averaged 2C DMRG over several roots vs 2C-FCI (RelCI)."""
    system = _na_system()

    scf = GHF(charge=0)(system)
    ci = CI(RelCISolver(nel=11, nroots=4, core_orbitals=10, active_orbitals=8))(scf)
    ci.run()

    scf2 = GHF(charge=0)(system)
    dmrg = CI(
        RelDMRGSolver(
            nel=11,
            nroots=4,
            core_orbitals=10,
            active_orbitals=8,
            dmrg_params=TIGHT(str(tmp_path)),
        )
    )(scf2)
    dmrg.run()

    assert len(dmrg.E_ci) == 4
    for i in range(4):
        assert dmrg.E_ci[i] == approx(ci.E_ci[i])
    assert dmrg.E == approx(0.25 * sum(ci.E_ci[:4]))


@requires_block2_complex
def test_reldmrg_sa_evals_per_solver_layout(tmp_path):
    """evals_per_solver has the single-state, multi-root shape like RelCISolver."""
    system = _na_system()
    scf = GHF(charge=0)(system)
    dmrg = CI(
        RelDMRGSolver(
            nel=11,
            nroots=4,
            core_orbitals=10,
            active_orbitals=8,
            dmrg_params=TIGHT(str(tmp_path)),
        )
    )(scf)
    dmrg.run()

    # Relativistic solvers use a single State (from nel) with multiple roots.
    evals_per_solver = dmrg.ci_solver.evals_per_solver
    assert len(dmrg.ci_solver.sub_solvers) == 1
    assert len(evals_per_solver) == 1
    assert len(evals_per_solver[0]) == 4
    assert np.allclose(np.concatenate(evals_per_solver), dmrg.E_ci)
