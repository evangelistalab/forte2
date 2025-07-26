import pytest
from forte2 import *
from forte2.helpers.comparisons import approx


def test_ci_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    cistate = CIStates(
        states=State(nel=2, multiplicity=1, ms=0.0), active_orbitals=[0, 1]
    )
    ci = CI(cistate)(rhf)
    ci.run()

    assert rhf.E == approx(-1.05643120731551)
    assert ci.E[0] == approx(-1.096071975854)


def test_ci_2():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_states = CIStates(
        states=State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
    )
    ci = CI(ci_states)(rhf)
    ci.run()

    assert ci.E[0] == approx(-100.019788438077)


def test_sa_ci_n2():
    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    singlet = State(14, multiplicity=1, ms=0.0)
    triplet = State(14, multiplicity=3, ms=0.0)
    sa_info = CIStates(
        states=[singlet, triplet],
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=[1, 2],
        weights=[[1.0], [0.85, 0.15]],
    )
    ci = CI(sa_info)(rhf)
    ci.run()
    eref_singlet = -109.004622061660
    eref_triplet1 = -108.779926502402
    eref_triplet2 = -108.733907910380
    assert ci.E[0] == approx(eref_singlet)
    assert ci.E[1] == approx(eref_triplet1)
    assert ci.E[2] == approx(eref_triplet2)
    assert ci.compute_average_energy() == approx(
        0.5 * eref_singlet + 0.5 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )


def test_sa_ci_with_avas():
    # This won't be strictly identical to test_sa_ci_n2 because AVAS will select different orbitals
    eref_singlet = -109.061384781871
    eref_triplet1 = -108.833136404913
    eref_triplet2 = -108.777400848037

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

    singlet = State(14, multiplicity=1, ms=0.0)
    triplet = State(14, multiplicity=3, ms=0.0)
    sa_info = CIStates(
        states=[singlet, triplet],
        avas=avas,
        nroots=[1, 2],
        weights=[[1.0], [0.85, 0.15]],
    )

    saci = CI(ci_states=sa_info, do_transition_dipole=True)(avas)
    saci.run()

    assert saci.E[0] == approx(eref_singlet)
    assert saci.E[1] == approx(eref_triplet1)
    assert saci.E[2] == approx(eref_triplet2)
    assert saci.compute_average_energy() == approx(
        0.5 * eref_singlet + 0.5 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )


def test_ci_tdm():
    xyz = f"""
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_states = CIStates(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=10,
    )
    ci = CI(ci_states, do_transition_dipole=True)(rhf)
    ci.run()
    assert abs(ci.tdm_per_solver[0][(0, 6)][2]) == approx(1.5435316739347478)
    assert ci.fosc_per_solver[0][(0, 6)] == approx(1.1589808047738437)
