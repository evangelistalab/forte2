import pytest

from forte2 import System, RHF, MCOptimizer, AVAS, AutoCI, MultiCI, State, CI
from forte2.helpers.comparisons import approx


def test_sa_mcscf_diff_mult_with_autoci():
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
    ci_singlet = AutoCI(charge=0, multiplicity=1, ms=0.0)
    ci_triplet = AutoCI(
        charge=0,
        multiplicity=3,
        ms=0.0,
        nroot=2,
        weights=[0.85, 0.15],
    )
    ci = MultiCI([ci_singlet, ci_triplet], weights=[0.25, 0.75])(avas)
    mc = MCOptimizer()(ci)
    mc.run()

    eref_singlet = -109.0664322107
    eref_triplet1 = -108.8450131892
    eref_triplet2 = -108.7888580871

    assert ci.E[0] == approx(eref_singlet)
    assert ci.E[1][0] == approx(eref_triplet1)
    assert ci.E[1][1] == approx(eref_triplet2)
    assert ci.E_avg[0] == approx(eref_singlet)
    assert ci.E_avg[1] == approx(0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    assert ci.compute_average_energy() == approx(
        0.25 * eref_singlet + 0.75 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )


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
    ci = AutoCI(charge=0, multiplicity=1, ms=0.0, nroot=3)(avas)
    mc = MCOptimizer()(ci)
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
    ci = CI(
        orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=2,
    )(rhf)
    mc = MCOptimizer(gconv=1e-7)(ci)
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
    ci_singlet = CI(
        core_orbitals=[0, 1, 2, 3],
        orbitals=[4, 5, 6, 7, 8, 9],
        state=State(14, multiplicity=1, ms=0.0),
        nroot=1,
    )
    ci_triplet = CI(
        core_orbitals=[0, 1, 2, 3],
        orbitals=[4, 5, 6, 7, 8, 9],
        state=State(14, multiplicity=3, ms=0.0),
        nroot=2,
        weights=[0.85, 0.15],
    )
    ci = MultiCI([ci_singlet, ci_triplet], weights=[0.25, 0.75])(rhf)
    mc = MCOptimizer()(ci)
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
    ci = CI(
        orbitals=[0, 1, 2, 3, 4, 5],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=4,
    )(rhf)
    mc = MCOptimizer(econv=1e-12, gconv=1e-12)(ci)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf_avg)
    assert mc.E_ci[0] == approx(emcscf_root_1)
    assert mc.E_ci[1] == approx(emcscf_root_2)
    assert mc.E_ci[2] == approx(emcscf_root_3)
    assert mc.E_ci[3] == approx(emcscf_root_4)
