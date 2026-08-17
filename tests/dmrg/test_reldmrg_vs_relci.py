"""
Relativistic (complex, two-component) DMRG tight-interface tests: energies.

These define the target API for `forte2.dmrg.RelDMRG` / `RelDMRGSolver`, which
must be drop-in compatible with `forte2.ci.RelCI` / `RelCISolver`. The
relativistic solver drives block2 in general-spin + complex mode
(SymmetryTypes.SGF | SymmetryTypes.CPX). With a large enough bond dimension,
DMRG on a small active space must reproduce the exact 2C-FCI energy from the
existing `RelCI` solver, computed in-test on the same spinors/active space.
"""

import pytest

from forte2 import System, RHF, GHF, SpinorUpcaster
from forte2.ci import RelCI
from forte2.dmrg import RelDMRG, RelDMRGSolver
from forte2.base_classes.params import DMRGParams, X2CParams
from forte2.base_classes import RelCIBase
from forte2.helpers.comparisons import approx

from conftest import requires_block2_complex

# Tight schedule so DMRG on a small active space is 2C-FCI-accurate.
TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[200] * 4 + [400] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-12] * 8,
    n_sweeps=12,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)


def _hf_system():
    return System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )


@requires_block2_complex
def test_reldmrg_orbital_rotation_invariant_flag_is_false():
    """RelDMRG is NOT exactly invariant to active-space (spinor) rotations,
    unlike exact 2C-FCI: a finite bond dimension is a basis-dependent
    truncation, so orbital_rotation_invariant must stay at RelCIBase's False
    default rather than opt in to True like RelCISolver does."""
    system = System(
        xyz="H 0.0 0.0 0.0",
        basis_set="sto-6g",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)
    dmrg = RelDMRG(nel=1, active_orbitals=2)(conv)
    assert not dmrg.orbital_rotation_invariant
    assert isinstance(dmrg, RelCIBase)
    assert dmrg.two_component


@requires_block2_complex
def test_reldmrg_vs_relci_h2(tmp_path):
    """H2/STO-6G, 4 active spinors: RelDMRG == 2C-RelCI ground state."""
    system = System(
        xyz="H 0.0 0.0 0.0\nH 0.0 0.0 2.0",
        basis_set="sto-6g",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)

    ci = RelCI(nel=2, active_orbitals=4)(conv)
    ci.run()

    dmrg = RelDMRG(nel=2, active_orbitals=4, dmrg_params=TIGHT(str(tmp_path)))(conv)
    dmrg.run()

    assert dmrg.two_component
    assert dmrg.E[0] == approx(ci.E[0])


@requires_block2_complex
def test_reldmrg_vs_relci_hf_x2c(tmp_path):
    """HF/cc-pVDZ with spin-orbit X2C, CAS(spinor): RelDMRG == 2C-RelCI."""
    system = System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )
    scf = GHF(charge=0)(system)

    ci = RelCI(nel=10, core_orbitals=2, active_orbitals=8)(scf)
    ci.run()

    scf2 = GHF(charge=0)(system)
    dmrg = RelDMRG(
        nel=10,
        core_orbitals=2,
        active_orbitals=8,
        dmrg_params=TIGHT(str(tmp_path)),
    )(scf2)
    dmrg.run()

    assert dmrg.E[0] == approx(ci.E[0])


@requires_block2_complex
def test_reldmrg_solver_is_relcibase_and_run_contract(tmp_path):
    """RelDMRGSolver must be a RelCIBase and expose the RelCISolver-like contract."""
    system = _hf_system()
    scf = RHF(charge=0, e_tol=1e-10)(system)
    conv = SpinorUpcaster(apply_random_phase=True)(scf)

    solver = RelDMRGSolver(
        nel=10,
        core_orbitals=2,
        active_orbitals=8,
        dmrg_params=TIGHT(str(tmp_path)),
    )
    assert isinstance(solver, RelCIBase)
    # Unlike RelCISolver, RelDMRGSolver does not opt in to
    # orbital_rotation_invariant: a finite bond dimension is not exactly
    # invariant to active-space rotations.
    assert not solver.orbital_rotation_invariant

    out = solver(conv)
    assert out is solver
    out = solver.run()
    assert out is solver

    assert solver.two_component
    assert len(solver.E) == 1
    assert len(solver.sub_solvers) == 1
    assert len(solver.evals_per_solver) == 1
    assert solver.get_convergence_status() == [True]
    assert solver.compute_average_energy() == approx(solver.E[0])
