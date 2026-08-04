"""
Relativistic DMRG tight-interface tests: 2C-DMRG-CASSCF via MCOptimizer.

Because RelDMRGSolver is a RelCIBase, it must plug into MCOptimizer exactly like
RelCISolver. On a small active space, 2C-DMRG-CASSCF must reproduce the 2C-CASSCF
energy obtained with the exact RelCI solver (computed in-test), exercising the
full set_ints / reset_eigensolver / make_average_{1,2}rdm contract across macro
iterations with complex integrals.
"""

import pytest

from forte2 import System, GHF, MCOptimizer, RelCISolver
from forte2.dmrg import RelDMRGSolver
from forte2.base_classes.params import DMRGParams
from forte2.helpers.comparisons import approx

from conftest import requires_block2_complex

# A tight DMRG schedule: on these small active spaces this is effectively exact,
# so the RDM truncation noise stays well below the CASSCF orbital-gradient
# tolerance and the macroiterations converge.
TIGHT = lambda scratch=None: DMRGParams(
    bond_dims=[400] * 4 + [800] * 4,
    noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
    thrds=[1e-14] * 8,
    n_sweeps=16,
    n_threads=1,
    iprint=0,
    scratch=scratch,
)

# DMRG-CASSCF: the orbital gradient can only be driven down to the DMRG RDM
# noise floor, so use a looser gradient threshold (and more macroiterations)
# than the exact-CI reference. The energy comparison itself stays at `approx`.
DMRG_MC_KWARGS = dict(g_tol=1e-6, maxiter=100)


@requires_block2_complex
@pytest.mark.slow
def test_reldmrg_casscf_hf_small_cas(tmp_path):
    """2C-DMRG-CASSCF == 2C-RelCI-CASSCF for HF (spin-orbit GHF), small CAS.

    Uses a spin-orbit GHF reference (a physical, non-degenerate 2C problem) so
    both solvers converge to the same minimum. A random-phase upcast of a
    spin-free RHF is deliberately avoided here: that system has a degenerate
    CASSCF solution manifold, and the exact-CI and DMRG optimizers can settle on
    different (equally valid) points, which makes an energy-equivalence
    assertion ill-posed.
    """
    system = System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c_type="so",
        snso_type=None,
    )
    scf = GHF(charge=0)(system)

    ci_ref = RelCISolver(nel=10, core_orbitals=4, active_orbitals=8)
    mc_ref = MCOptimizer(ci_ref, g_tol=1e-7)(scf)
    mc_ref.run()

    scf2 = GHF(charge=0)(system)
    dmrg_solver = RelDMRGSolver(
        nel=10, core_orbitals=4, active_orbitals=8, dmrg_params=TIGHT(str(tmp_path))
    )
    mc = MCOptimizer(dmrg_solver, **DMRG_MC_KWARGS)(scf2)
    mc.run()

    assert mc.E == approx(mc_ref.E)


@requires_block2_complex
@pytest.mark.slow
def test_reldmrg_casscf_hf_ghf(tmp_path):
    """2C-DMRG-CASSCF == 2C-RelCI-CASSCF for a spin-orbit GHF reference."""
    system = System(
        xyz="H 0.0 0.0 0.0\nF 0.0 0.0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c_type="so",
        snso_type=None,
    )
    scf = GHF(charge=0)(system)

    ci_ref = RelCISolver(nel=10, core_orbitals=2, active_orbitals=12)
    mc_ref = MCOptimizer(ci_ref, g_tol=1e-7)(scf)
    mc_ref.run()

    scf2 = GHF(charge=0)(system)
    dmrg_solver = RelDMRGSolver(
        nel=10, core_orbitals=2, active_orbitals=12, dmrg_params=TIGHT(str(tmp_path))
    )
    mc = MCOptimizer(dmrg_solver, **DMRG_MC_KWARGS)(scf2)
    mc.run()

    assert mc.E == approx(mc_ref.E)
