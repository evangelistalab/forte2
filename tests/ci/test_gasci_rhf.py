from numpy import isclose # type: ignore

from forte2 import *


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

    assert isclose(rhf.E, -1.05643120731551)
    assert isclose(ci.E[0], -1.096071975854)


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

    assert isclose(rhf.E, -1.089283671174)
    assert isclose(ci.E[0], -1.089283671174)
    # TODO: Add assertion for second root when the one below is externally verified
    # assert isclose(ci.E[1], -0.671622137375)

def test_gasci_rhf_3():
    xyz = f"""
    H  0.000000000000  0.000000000000 -0.375000000000
    H  0.000000000000  0.000000000000  0.375000000000
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0], [1]],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0,0],
        gas_max=[2,2],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -1.12475114835983)
    assert isclose(ci.E[0], -1.145766051194)

def test_gasci_rhf_4():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="6-31G**", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0,1,2,3,4], [5,6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[7,0],
        gas_max=[10,4],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -76.01726423763337)
    assert isclose(ci.E[0], -76.030899030220)
    # Fails, CI energy does not match RDM energy

def test_gasci_rhf_5():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="6-31G**", auxiliary_basis="cc-pvdz-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0], [1,2,3,4,5,6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -76.01725998416885)
    assert isclose(ci.E[0], -55.841523397373)
    # Exception: [forte2] Basis Set Exchange does not have data for element Z=8 in basis set cc-pvdz-jkfit!

def test_gasci_rhf_6():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="6-31G**", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0], [1,2,3,4,5,6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        gas_min=[0],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -76.01726423763337)
    assert isclose(ci.E[0], -55.841496017363)
    # Fails, CI energy does not match RDM energy

def test_gasci_rhf_7():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="6-31G**", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0], [1,2,3,4,5,6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=2,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -76.01726423763337)
    assert isclose(ci.E[0], -55.840353918103)
    assert isclose(ci.E[1], -55.767946273823) # fails


def test_gasci_rhf_8():
    xyz = f"""
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="def2-universal-jkfit")

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0], [1,2,3,4,5,6]],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=2,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -75.68026686304654)
    assert isclose(ci.E[0], -55.598001374143)
    assert isclose(ci.E[1], -55.525624692892)


def test_gasci_rohf_1():
    xyz = f"""
    C           -0.055505285387     0.281253495230     0.333445183956
    H            1.241755444477     0.529009702423     1.928202188113
    H            0.960863251201    -0.259845503299    -1.606568704186
    H           -1.849273709593     1.285938489070     0.087436138153
    H           -0.035109236026    -1.797124531812    -0.544640316685
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="def2-universal-jkfit", unit="bohr")

    rhf = ROHF(charge=1, econv=1e-12, dconv=1e-8, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0], [1,2,3,4,5,6,7,8]],
        state=State(nel=9, multiplicity=2, ms=0.5),
        nroot=1,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -39.66353334247484)
    assert isclose(ci.E[0], -29.237219037891)
    # rohf energy does not match

def test_gasci_rohf_2():
    xyz = f"""
    C           -0.055505285387     0.281253495230     0.333445183956
    H            1.241755444477     0.529009702423     1.928202188113
    H            0.960863251201    -0.259845503299    -1.606568704186
    H           -1.849273709593     1.285938489070     0.087436138153
    H           -0.035109236026    -1.797124531812    -0.544640316685
    """

    system = System(xyz=xyz, basis="cc-pcvtz", auxiliary_basis="def2-universal-jkfit", unit="bohr")

    rhf = ROHF(charge=1, econv=1e-12, dconv=1e-8, guess_type='hcore')(system)
    ci = CI(
        orbitals=[[0], [1,2,3,4,5,6,7,8]],
        state=State(nel=9, multiplicity=2, ms=0.5),
        nroot=1,
        gas_min=[1],
        gas_max=[1],
    )(rhf)
    ci.run()

    assert isclose(rhf.E, -39.7798679697753528)
    assert isclose(ci.E[0], -29.205450125466)
    # Exception: [forte2] Basis set cc-pcvtz does not contain element 1.