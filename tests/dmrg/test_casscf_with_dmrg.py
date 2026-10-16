import pytest

from forte2 import System, RHF, State, MCOptimizer, CISolver
from forte2.dmrg import DMRGSolver
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


@requires_block2
@pytest.mark.slow
def test_dmrg_casscf_n2_ground_state(tmp_path):
    """CAS(6,6) DMRG-CASSCF ground state == CI-CASSCF ground state for N2."""
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.4",
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    hf = RHF(charge=0, e_tol=1e-12)(system)
    ci_ref = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
    )
    mc_ref = MCOptimizer(ci_ref, g_tol=1e-7)(hf)
    mc_ref.run()

    hf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg_solver = DMRGSolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
    )
    mc = MCOptimizer(dmrg_solver, g_tol=1e-7)(hf)
    mc.run()

    assert mc.E == approx(mc_ref.E)


@requires_block2
@pytest.mark.slow
def test_dmrg_sa_casscf_n2(tmp_path):
    """State-averaged (2-root) DMRG-CASSCF == CI-CASSCF for N2."""
    system = System(
        xyz="N 0.0 0.0 0.0\nN 0.0 0.0 1.4",
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    hf = RHF(charge=0, e_tol=1e-12)(system)
    ci_ref = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        nroots=2,
    )
    mc_ref = MCOptimizer(ci_ref, g_tol=1e-7)(hf)
    mc_ref.run()

    hf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg_solver = DMRGSolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        nroots=2,
        dmrg_params=TIGHT_DMRG_PARAMS(str(tmp_path)),
    )
    mc = MCOptimizer(dmrg_solver, g_tol=1e-7)(hf)
    mc.run()

    assert mc.E_ci[0] == approx(mc_ref.E_ci[0])
    assert mc.E_ci[1] == approx(mc_ref.E_ci[1])
    assert mc.E == approx(mc_ref.E)
