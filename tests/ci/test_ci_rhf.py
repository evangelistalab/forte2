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


def test_multici_1():
    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis="cc-pvdz", auxiliary_basis="cc-pVTZ-JKFIT")

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
    ci.run()
    eref_singlet = -109.004622061660
    eref_triplet1 = -108.779926502402
    eref_triplet2 = -108.733907910380
    assert ci.E[0] == approx(eref_singlet)
    assert ci.E[1][0] == approx(eref_triplet1)
    assert ci.E[1][1] == approx(eref_triplet2)
    assert ci.E_avg[0] == approx(eref_singlet)
    assert ci.E_avg[1] == approx(0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    assert ci.compute_average_energy() == approx(
        0.25 * eref_singlet + 0.75 * (0.85 * eref_triplet1 + 0.15 * eref_triplet2)
    )
