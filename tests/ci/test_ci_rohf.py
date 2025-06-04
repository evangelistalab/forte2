from numpy import isclose

from forte2 import *


def test_rohf_ci_1():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = ROHF(charge=1, ms=0.5, econv=1e-12)(system)
    ci = CI(
        orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        state=State(nel=9, multiplicity=2, ms=0.5),
        nroot=2,
    )(rhf)
    ci.run()

    assert isclose(ci.E[0], -99.510706628367)


def test_rohf_ci_2():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = ROHF(charge=1, ms=-0.5, econv=1e-12)(system)
    ci = CI(
        orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        state=State(nel=9, multiplicity=2, ms=-0.5),
        nroot=2,
    )(rhf)
    ci.run()

    assert isclose(ci.E[0], -99.510706628367)


if __name__ == "__main__":
    test_rohf_ci_1()
    test_rohf_ci_2()
