"""
DMRG (block2) tight-interface tests: energies vs. exact FCI.

These tests define the target API for `forte2.dmrg.DMRG` / `DMRGSolver`, which
must be drop-in compatible with `forte2.ci.CI` / `CISolver`. With a large enough
bond dimension, DMRG on a small active space must reproduce the FCI energy from
the existing `CI` solver, computed in-test on the same orbitals/active space.
"""

import pytest

from forte2 import System, RHF, State, CI
from forte2.dmrg import DMRG, DMRGSolver
from forte2.base_classes.params import DMRGParams
from forte2.base_classes import CIBase
from forte2.helpers.comparisons import approx

from conftest import requires_block2

# A tight schedule so DMRG on a small CAS is FCI-accurate.
TIGHT_DMRG_PARAMS = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=12,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


@requires_block2
def test_dmrg_orbital_invariance_is_true():
    """DMRG is invariant to active-space orbital rotations (like FCI)."""
    xyz = "H 0.0 0.0 0.0"
    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    hf = RHF(charge=-1, e_tol=1e-12)(system)
    dmrg = DMRG(
        State(system=system, multiplicity=2, ms=0.5),
        active_orbitals=[0, 1],
    )(hf)
    assert dmrg.orbital_rotation_invariant
    assert isinstance(dmrg, CIBase)


@requires_block2
def test_dmrg_vs_fci_h2(tmp_path):
    """H2/STO-6G, CAS(2,2): DMRG == FCI ground-state energy."""
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.058354421806
    """
    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    state = State(nel=2, multiplicity=1, ms=0.0)
    ci = CI(states=state, active_orbitals=[0, 1])(rhf)
    ci.run()

    rhf2 = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=state,
        active_orbitals=[0, 1],
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
    )(rhf2)
    dmrg.run()

    assert dmrg.E[0] == approx(ci.E[0])


@requires_block2
def test_dmrg_vs_fci_n2_cas66(tmp_path):
    """N2/cc-pVDZ, CAS(6,6): DMRG == FCI ground-state energy."""
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    state = State(nel=14, multiplicity=1, ms=0.0)
    ci = CI(
        states=state,
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
    )(rhf)
    ci.run()

    rhf2 = RHF(charge=0, e_tol=1e-12)(system)
    dmrg = DMRG(
        states=state,
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
    )(rhf2)
    dmrg.run()

    assert dmrg.E[0] == approx(ci.E[0])


@requires_block2
def test_dmrg_solver_is_cibase_and_run_returns_self(tmp_path):
    """DMRGSolver must be a CIBase and expose the CISolver-like run contract."""
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.058354421806
    """
    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    solver = DMRGSolver(
        states=State(nel=2, multiplicity=1, ms=0.0),
        active_orbitals=[0, 1],
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
    )
    assert isinstance(solver, CIBase)
    assert solver.orbital_rotation_invariant

    out = solver(rhf)
    assert out is solver
    out = solver.run()
    assert out is solver

    # Standard post-run attributes provided by CISolver-like solvers.
    assert len(solver.E) == 1
    assert len(solver.sub_solvers) == 1
    assert len(solver.evals_per_solver) == 1
    assert solver.get_convergence_status() == [True]
    assert solver.compute_average_energy() == approx(solver.E[0])
