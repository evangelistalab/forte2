import pytest
import numpy as np
from forte2 import System, RHF, MCOptimizer, AVAS, CIStates, State, CI
from forte2.helpers.comparisons import approx


def test_sa_mcscf_diff_mult_with_avas():
    # This should be strictly identical to test_mcscf_sa_diff_mult given a sufficiently robust MCSCF solver.
    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=3,
        num_active_uocc=3,
        subspace=["N(2p)"],
        diagonalize=True,
    )(rhf)
    singlet = State(nel=rhf.nel, multiplicity=1, ms=0.0)
    triplet = State(nel=rhf.nel, multiplicity=3, ms=0.0)
    ci_states = CIStates(
        states=[singlet, triplet],
        avas=avas,
        weights=[[0.25], [0.75 * 0.85, 0.75 * 0.15]],
        nroots=[1, 2],
    )
    mc = MCOptimizer(ci_states)(avas)
    mc.run()

    eref_singlet = -109.0664322107
    eref_triplet1 = -108.8450131892
    eref_triplet2 = -108.7888580871

    assert mc.E_ci[0] == approx(eref_singlet)
    assert mc.E_ci[1] == approx(eref_triplet1)
    assert mc.E_ci[2] == approx(eref_triplet2)
    assert mc.ci_solver.compute_average_energy() == approx(
        0.25 * eref_singlet + 0.75 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )

    nat_occ_ref = np.array(
        [
            [1.97531263, 1.97618291, 1.9797172],
            [1.91200525, 1.45253886, 1.46608634],
            [1.91200525, 1.45253876, 1.46608622],
            [0.08779585, 0.54735532, 0.53380309],
            [0.08779585, 0.54735522, 0.53380298],
            [0.02508518, 0.02402893, 0.02050418],
        ]
    )
    assert mc.ci_solver.nat_occs == pytest.approx(nat_occ_ref, abs=5e-7)


def test_sa_casscf_c2():
    """Test CASSCF with C2 molecule."""

    erhf = -75.382486499716
    ecasscf = -75.5580517997

    xyz = f"""
    C 0.0 0.0 0.0
    C 0.0 0.0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=4,
        num_active_uocc=4,
        subspace=["C(2s)", "C(2p)"],
    )(rhf)
    ci_state = CIStates(
        states=State(nel=rhf.nel, multiplicity=1, ms=0.0), avas=avas, nroots=3
    )
    mc = MCOptimizer(ci_state)(avas)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_sa_casscf_same_mult():
    erhf = -108.761639873604
    ecasscf = -108.8592663803

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        states=State(nel=14, multiplicity=1, ms=0.0),
        nroots=2,
    )
    mc = MCOptimizer(ci_state, gconv=1e-7)(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_sa_casscf_diff_mult():
    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    singlet = State(nel=rhf.nel, multiplicity=1, ms=0.0)
    triplet = State(nel=rhf.nel, multiplicity=3, ms=0.0)
    ci_states = CIStates(
        states=[singlet, triplet],
        active_spaces=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        weights=[[0.25], [0.75 * 0.85, 0.75 * 0.15]],
        nroots=[1, 2],
    )
    mc = MCOptimizer(ci_states)(rhf)
    mc.run()

    eref_singlet = -109.0664322107
    eref_triplet1 = -108.8450131892
    eref_triplet2 = -108.7888580871

    assert mc.E_ci[0] == approx(eref_singlet)
    assert mc.E_ci[1] == approx(eref_triplet1)
    assert mc.E_ci[2] == approx(eref_triplet2)
    assert mc.E == approx(
        0.25 * eref_singlet + 0.75 * (eref_triplet1 * 0.85 + eref_triplet2 * 0.15)
    )


def test_sa_casscf_hf():
    erhf = -100.00987356244831
    emcscf_root_1 = -99.996420746310
    emcscf_root_2 = -99.688682892330
    emcscf_root_3 = -99.688682892330
    emcscf_root_4 = -99.470229157315
    emcscf_avg = -99.71100392207127

    xyz = f"""
    H            0.000000000000     0.000000000000    -0.949624435830
    F            0.000000000000     0.000000000000     0.050375564170
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_state = CIStates(
        active_spaces=[0, 1, 2, 3, 4, 5],
        states=State(nel=10, multiplicity=1, ms=0.0),
        nroots=4,
    )
    mc = MCOptimizer(ci_state, econv=1e-12, gconv=1e-12, do_transition_dipole=True)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf_avg)
    assert mc.E_ci[0] == approx(emcscf_root_1)
    assert mc.E_ci[1] == approx(emcscf_root_2)
    assert mc.E_ci[2] == approx(emcscf_root_3)
    assert mc.E_ci[3] == approx(emcscf_root_4)

    assert abs(mc.ci_solver.tdm_per_solver[0][(0, 3)][2]) == approx(1.1358091937504018)
    assert mc.ci_solver.fosc_per_solver[0][(0, 3)] == approx(0.4525467009634354)


def test_sa_casscf_c2_transition_dipole():
    """Test CASSCF with C2 molecule."""

    erhf = -75.382486499716
    ecasscf = -75.5580517997

    xyz = f"""
    C 0.0 0.0 0.0
    C 0.0 0.0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    singlet = State(nel=rhf.nel, multiplicity=1, ms=0.0)
    triplet = State(nel=rhf.nel, multiplicity=3, ms=0.0)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=4,
        num_active_uocc=4,
        subspace=["C(2s)", "C(2p)"],
    )(rhf)
    ci_state = CIStates(states=[singlet, triplet], avas=avas, nroots=[2, 2])
    mc = MCOptimizer(ci_state, do_transition_dipole=True)(avas)
    mc.run()
    assert mc.ci_solver.evals_per_solver[0][0] == approx(-75.6107427252)
    assert mc.ci_solver.evals_per_solver[0][1] == approx(-75.5314535451)
    assert mc.ci_solver.evals_per_solver[1][0] == approx(-75.5792187010)
    assert mc.ci_solver.evals_per_solver[1][1] == approx(-75.5789867708)
    assert mc.ci_solver.fosc_per_solver[0][(0, 1)] == approx(0.006525243082121279)
