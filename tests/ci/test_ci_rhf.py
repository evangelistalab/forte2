from forte2 import *
from forte2.helpers.comparisons import approx


def test_ci_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis="sto-6g", auxiliary_basis="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[0, 1],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.05643120731551)
    assert ci.E[0] == approx(-1.096071975854)


def test_ci_2():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    ci.run()

    assert ci.E[0] == approx(-100.019788438077)
