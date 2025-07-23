import pytest
from forte2 import *
from forte2.helpers.comparisons import approx


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
    ci_state = CIStates(
        active_spaces=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        states=State(nel=9, multiplicity=2, ms=0.5, gas_min=[1], gas_max=[1]),
    )
    ci = CI(ci_state, econv=1e-12)(rhf)
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
    ci_state = CIStates(
        active_spaces=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
        states=State(nel=9, multiplicity=2, ms=0.5, gas_min=[1], gas_max=[1]),
    )
    ci = CI(ci_state, econv=1e-12)(rhf)
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
    ci_state = CIStates(
        active_spaces=[[0], [1, 2, 3, 4, 5, 6]],
        states=State(nel=10, multiplicity=3, ms=1.0, gas_min=[0], gas_max=[1]),
    )
    ci = CI(ci_state)(rhf)
    ci.run()

    assert rhf.E == approx(-75.78642207312076)
    assert ci.E[0] == approx(-56.130750582569)
