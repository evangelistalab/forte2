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
        orbitals=[[0], [1]],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0],
        gas_max=[2],
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
    # assert ci.E[1] == approx(-0.671622137375)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="Basis set exchange is not available")
def test_gasci_rhf_3():
    xyz = f"""
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
        orbitals=[[0], [1]],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0, 0],
        gas_max=[2, 2],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.124751148359)
    assert ci.E[0] == approx(-1.145766051194)


def test_gasci_rhf_4():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        orbitals=[[0, 1, 2, 3, 4], [5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[6, 0],
        gas_max=[10, 4],
        econv=1e-12,
        # orbitals=[[0, 1, 2, 3, 4, 5, 6]],
        # state=State(nel=10, multiplicity=1, ms=0.0),
        # nroot=1,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-76.029447292783)


def test_gasci_rhf_5():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-14, dconv=1e-8)(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0],
        gas_max=[1],
        econv=1e-14,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-55.819535370117)


def test_gasci_rhf_6():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        orbitals=[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[4, 0, 0],
        gas_max=[6, 8, 2],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-100.00984797870581)
    assert ci.E[0] == approx(-100.088791333620)


def test_gasci_rhf_7():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=2,
        gas_min=[1],
        gas_max=[1],
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-55.818614251158)
    assert ci.E[1] == approx(-55.740542172622)


def test_gasci_rhf_8():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=2,
        gas_min=[1],
        gas_max=[1],
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-75.68026686304654)
    assert ci.E[0] == approx(-55.598443621487)
    assert ci.E[1] == approx(-55.526088426266)


def test_gasci_rohf_1():
    xyz = f"""
    C           -0.055505285387     0.281253495230     0.333445183956
    H            1.241755444477     0.529009702423     1.928202188113
    H            0.960863251201    -0.259845503299    -1.606568704186
    H           -1.849273709593     1.285938489070     0.087436138153
    H           -0.035109236026    -1.797124531812    -0.544640316685
    """

    system = System(
        xyz=xyz,
        basis_set="sto-6g",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = ROHF(charge=1, ms=0.5, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        state=State(nel=9, multiplicity=2, ms=0.5),
        nroot=1,
        gas_min=[1],
        gas_max=[1],
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-39.66353334247423)
    assert ci.E[0] == approx(-29.237267496782)


def test_gasci_rohf_2():
    xyz = f"""
    C           -0.055505285387     0.281253495230     0.333445183956
    H            1.241755444477     0.529009702423     1.928202188113
    H            0.960863251201    -0.259845503299    -1.606568704186
    H           -1.849273709593     1.285938489070     0.087436138153
    H           -0.035109236026    -1.797124531812    -0.544640316685
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvtz",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = ROHF(charge=1, ms=0.5, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        state=State(nel=9, multiplicity=2, ms=0.5),
        nroot=1,
        gas_min=[1],
        gas_max=[1],
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-39.779741004794)
    assert ci.E[0] == approx(-29.204808393068)


def test_gasci_rohf_3():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = ROHF(charge=0, econv=1e-12, dconv=1e-8, ms=1.0)(system)
    ci = CI(
        orbitals=[[0], [1, 2, 3, 4, 5, 6]],
        state=State(nel=10, multiplicity=3, ms=1.0),
        nroot=1,
        gas_min=[0],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert rhf.E == approx(-75.78642207312076)
    assert ci.E[0] == approx(-56.130750582569)
