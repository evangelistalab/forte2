import numpy as np
import pytest

from forte2 import CI, CISolver, DMRGSolver, MCOptimizer, RHF, ROHF, State, System
from forte2.base_classes import DMRGParams
from forte2.dmrg import BLOCK2_AVAILABLE
from forte2.dsrg import DSRG_MRPT2
from forte2.helpers.comparisons import approx


def test_sf_mrpt2_n2():
    erhf = -108.954140898736
    emcscf = -109.0811491968

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0)(system)
    rhf.run()

    ci_solver = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=4,
        active_orbitals=6,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="iterate")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(-109.23886074061)
    assert dsrg.relax_energies[0, 1] == approx(-109.23931193044)
    assert dsrg.relax_energies[0, 2] == approx(-109.08114919682)

    assert dsrg.relax_energies[1, 0] == approx(-109.23895207574)
    assert dsrg.relax_energies[1, 1] == approx(-109.23895208449)
    assert dsrg.relax_energies[1, 2] == approx(-109.08065641191)

    assert dsrg.relax_energies[2, 0] == approx(-109.23895388557)
    assert dsrg.relax_energies[2, 1] == approx(-109.23895388557)
    assert dsrg.relax_energies[2, 2] == approx(-109.08065911063)


def test_sf_mrpt2_o2_triplet():
    xyz = """
    O 0.0 0.0 0.0
    O 0.0 0.0 1.251
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    mf = ROHF(charge=0, ms=1.0)(system)
    ci_solver = CISolver(
        states=State(nel=16, multiplicity=3, ms=1.0),
        core_orbitals=4,
        active_orbitals=6,
    )
    mc = MCOptimizer(ci_solver)(mf)
    dsrg = DSRG_MRPT2(
        flow_param=1.0,
        relax_reference="twice",
        frozen_core_orbitals=2,
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 0] == approx(-149.963322736742)
    assert dsrg.relax_energies[0, 1] == approx(-149.964938525217)
    assert dsrg.relax_energies[0, 2] == approx(-149.707577515944)
    assert dsrg.relax_energies[1, 0] == approx(-149.96533377181)
    assert dsrg.relax_energies[1, 1] == approx(-149.965334494277)
    assert dsrg.relax_energies[1, 2] == approx(-149.70550603407)

    # diagonalizing hbar should reproduce most recent relaxed energy
    hbar0 = dsrg.hbar0
    hbar1 = dsrg.hbar1_canon
    hbar2 = dsrg.hbar2_canon
    ci_solver.set_ints(hbar0, hbar1, hbar2)
    ci_solver.run()
    assert ci_solver.E[0] == approx(-149.965334494277)  # dsrg.relax_energies[1, 1]


def test_mrpt2_all_active():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(State(nel=2, multiplicity=1, ms=0.0), active_orbitals=2)
    mc = MCOptimizer(
        ci_solver,
        maxiter=5,
    )(rhf)
    pt = DSRG_MRPT2(flow_param=0.5)(mc)
    pt.run()
    assert pt.E_dsrg == approx(-1.096071975854)


def test_mrpt2_c_atom_ci_reference_relaxation():
    """
    Reference relaxation with a CI reference
    """
    escf = -37.682415354711
    edsrg = -37.741130215328
    edsrg_relaxed = -37.741196308282
    eref = -37.678689834640

    system = System(
        xyz="C 0 0 0",
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    scf = ROHF(charge=0, ms=1.0)(system)
    ci = CI(
        CISolver(
            State(nel=6, multiplicity=3, ms=1.0),
            nroots=3,
            core_orbitals=1,
            active_orbitals=4,
        ),
        final_orbitals="semicanonical",
    )(scf)
    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(ci)
    dsrg.run()

    assert scf.E == approx(escf)
    assert dsrg.relax_energies[0, 0] == approx(edsrg)
    assert dsrg.relax_energies[0, 1] == approx(edsrg_relaxed)
    assert dsrg.relax_energies[0, 2] == approx(eref)


@pytest.mark.skipif(not BLOCK2_AVAILABLE, reason="block2 is not installed")
def test_dmrg_dsrg_n2_relax(tmp_path):
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.1
    """
    state = State(nel=14, multiplicity=1, ms=0.0)
    core_orbitals = [0, 1, 2, 3]
    active_orbitals = [4, 5, 6, 7, 8, 9]

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        states=state, core_orbitals=core_orbitals, active_orbitals=active_orbitals
    )
    mc = MCOptimizer(ci_solver)(rhf)
    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="twice")(mc)
    dsrg.run()

    # Six sites, so a bond dimension of a few hundred is exact, and perturbative noise
    # can't move a solve off an excited state it lands on
    dmrg_params = DMRGParams(
        bond_dims=[200] * 4 + [400] * 4,
        noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
        noise_type="DensityMatrix",
        davidson_tols=[1e-12] * 8,
        n_sweeps=12,
        orbital_ordering="gaopt",
        scratch=str(tmp_path),
    )
    dmrg_solver = DMRGSolver(
        states=state,
        core_orbitals=core_orbitals,
        active_orbitals=active_orbitals,
        dmrg_params=dmrg_params,
    )
    dmrgscf = MCOptimizer(dmrg_solver)(rhf)
    # DSRG-MRPT2 needs the 3-cumulant, and each relaxation restarts the DMRG
    dmrg_dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="twice")(dmrgscf)
    dmrg_dsrg.run()

    assert dmrg_solver.get_convergence_status() == [True]
    assert dmrgscf.E_ci[0] == approx(mc.E_ci[0])
    np.testing.assert_allclose(
        dmrg_dsrg.relax_energies, dsrg.relax_energies, rtol=0, atol=5e-8
    )
