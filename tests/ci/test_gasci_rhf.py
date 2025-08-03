import pytest
from forte2 import *
from forte2.helpers.comparisons import approx
from forte2.system.build_basis import BSE_AVAILABLE


def test_gasci_rhf_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        active_orbitals=[[0], [1]],
        states=State(nel=2, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[2]),
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.05643120731551)
    assert ci.E[0] == approx(-1.096071975854)


def test_gasci_rhf_2():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        active_orbitals=[[0], [1]],
        states=State(nel=2, multiplicity=1, ms=0.0, gas_min=[1, 0], gas_max=[2, 1]),
        nroots=2,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.089283671174)
    assert ci.E[0] == approx(-1.089283671174)
    # TODO: Add assertion for second root when the one below is externally verified
    # assert ci.E[1] == approx(-0.671622137375)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="Basis set exchange is not available")
def test_gasci_rhf_3():
    xyz = """
    H  0.000000000000  0.000000000000 -0.375000000000
    H  0.000000000000  0.000000000000  0.375000000000
    """

    system = System(
        xyz=xyz,
        basis_set="sto-6g",
        auxiliary_basis_set="def2-universal-jkfit",
        auxiliary_basis_set_corr="def2-svp-rifit",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        active_orbitals=[[0], [1]],
        states=State(nel=2, multiplicity=1, ms=0.0, gas_min=[0, 0], gas_max=[2, 2]),
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.124751148359)
    assert ci.E[0] == approx(-1.145766051194)


def test_gasci_rhf_4():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        active_orbitals=[[0, 1, 2, 3, 4], [5, 6]],
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[6, 0], gas_max=[10, 4]),
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-76.029447292783)


def test_gasci_rhf_5():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-14, dconv=1e-8)(system)
    ci = CI(
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[1]),
        econv=1e-14,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-55.819535370117)


def test_gasci_rhf_6():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        active_orbitals=(3, 4, 3),
        states=State(
            nel=10, multiplicity=1, ms=0.0, gas_min=[4, 0, 0], gas_max=[6, 8, 2]
        ),
    )(rhf)
    ci.run()

    assert rhf.E == approx(-100.00984797870581)
    assert ci.E[0] == approx(-100.088791333620)


def test_gasci_rhf_7():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
        nroots=2,
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-55.818614251158)
    assert ci.E[1] == approx(-55.740542172622)


def test_gasci_rhf_8():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
        nroots=2,
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-75.68026686304654)
    assert ci.E[0] == approx(-55.598443621487)
    assert ci.E[1] == approx(-55.526088426266)
