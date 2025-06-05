from numpy import isclose

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


if __name__ == "__main__":
    test_gasci_rhf_1()
    test_gasci_rhf_2()
