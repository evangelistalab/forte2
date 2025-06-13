import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_gasci_rhf_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[[0], [1]],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0],
        gas_max=[2],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.05643120731551)
    assert ci.E[0] == approx(-1.096071975854)


def test_gasci_rhf_2():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[[0], [1]],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=2,
        gas_min=[1, 0],
        gas_max=[2, 1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.089283671174)
    assert ci.E[0] == approx(-1.089283671174)
    # TODO: Add assertion for second root when the one below is externally verified
    # assert isclose(ci.E[1], -0.671622137375)


def test_gasci_rhf_3():
    xyz = f"""
    H  0.000000000000  0.000000000000 -0.375000000000
    H  0.000000000000  0.000000000000  0.375000000000
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0], [1]],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0, 0],
        gas_max=[2, 2],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.12475114835983)
    assert ci.E[0] == approx(-1.145763462077)


@pytest.mark.xfail(reason="CI energy does not match RDM energy")
def test_gasci_rhf_4():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0, 1, 2, 3, 4], [5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[6, 0],
        gas_max=[10, 4],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-76.030555835340)


@pytest.mark.xfail(reason="Fails, CI energy does not match RDM energy")
def test_gasci_rhf_5():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-55.818855513012)


@pytest.mark.xfail(reason="uhf energy does not match reference value")
def test_gasci_rhf_6():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit")

    rhf = UHF(charge=0, econv=1e-12, dconv=1e-8, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=3, ms=1.0),
        nroot=1,
        gas_min=[0],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-75.70155175095266)
    assert ci.E[0] == approx(-56.129450806753)


@pytest.mark.xfail(reason="CI root 1 does not match reference value")
def test_gasci_rhf_7():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=2,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-55.817934328246)
    assert ci.E[1] == approx(-55.740177190272)


def test_gasci_rhf_8():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=2,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-75.68026686304654)
    assert ci.E[0] == approx(-55.598001374143)
    assert ci.E[1] == approx(-55.525624692892)


@pytest.mark.xfail(reason="rohf energy does not match reference value")
def test_gasci_rohf_1():
    xyz = f"""
    C           -0.055505285387     0.281253495230     0.333445183956
    H            1.241755444477     0.529009702423     1.928202188113
    H            0.960863251201    -0.259845503299    -1.606568704186
    H           -1.849273709593     1.285938489070     0.087436138153
    H           -0.035109236026    -1.797124531812    -0.544640316685
    """

    system = System(
        xyz=xyz, basis="sto-6g", auxiliary_basis="def2-universal-jkfit", unit="bohr"
    )

    rhf = ROHF(charge=1, econv=1e-12, dconv=1e-8, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        state=State(nel=9, multiplicity=2, ms=0.5),
        nroot=1,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-39.66353334247484)
    assert ci.E[0] == approx(-29.237219037891)


@pytest.mark.xfail(reason="rohf energy does not match reference value")
def test_gasci_rohf_2():
    xyz = f"""
    C           -0.055505285387     0.281253495230     0.333445183956
    H            1.241755444477     0.529009702423     1.928202188113
    H            0.960863251201    -0.259845503299    -1.606568704186
    H           -1.849273709593     1.285938489070     0.087436138153
    H           -0.035109236026    -1.797124531812    -0.544640316685
    """

    system = System(
        xyz=xyz, basis="cc-pvtz", auxiliary_basis="def2-universal-jkfit", unit="bohr"
    )

    rhf = ROHF(charge=1, econv=1e-12, dconv=1e-8, guess_type="hcore")(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        state=State(nel=9, multiplicity=2, ms=0.5),
        nroot=1,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-39.77974100479403)
    assert ci.E[0] == approx(-29.204823485711)


test_gasci_rhf_3()
