import pytest

from forte2 import *
from forte2.helpers.comparisons import approx


def test_mcscf_1():
    erhf = -1.08928367118043
    emcscf = -1.11873740345286

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[0, 1],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_mcscf_2():
    erhf = -99.9977252002946
    emcscf = -100.0435018956

    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.maxiter = 200
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_mcscf_3():
    erhf = -108.761639873604
    ecasscf = -108.9800484156

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.gradtol = 1e-7
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_mcscf_sa_same_mult():
    erhf = -108.761639873604
    ecasscf = -108.8592663803

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        orbitals=[4, 5, 6, 7, 8, 9],
        core_orbitals=[0, 1, 2, 3],
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=2,
    )(rhf)
    mc = MCOptimizer()(ci)
    mc.gradtol = 1e-7
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)


def test_mcscf_sa_diff_mult():
    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.2
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")

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
    mc = MCOptimizer()(ci)
    mc.run()

    eref_singlet = -109.0664322107
    eref_triplet1 = -108.8450131892
    eref_triplet2 = -108.7888580871

    assert mc.E_ci[0] == approx(eref_singlet)
    assert mc.E_ci[1] == approx(eref_triplet1)
    assert mc.E_ci[2] == approx(eref_triplet2)
    assert mc.E == approx(
        0.25 * eref_singlet + 0.75 * (eref_triplet1 * 0.85 + eref_triplet2 * 0.15)
    )


def test_mcscf_noncontiguous_spaces():
    # The results of this test should be strictly identical to test_mcscf_3
    import numpy as np

    erhf = -108.761639873604
    eci = -108.916505576963
    ecasscf = -108.9800484156

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.4
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf.run()
    assert rhf.E == approx(erhf)

    # swap orbitals to make them non-contiguous
    core = [0, 1, 3, 6]
    actv = [2, 4, 5, 7, 8, 11]
    virt = sorted(set(range(system.nbf)) - set(core + actv))
    rhf.C[0][:, core + actv + virt] = rhf.C[0]
    ci = CI(
        orbitals=actv,
        core_orbitals=core,
        state=State(nel=14, multiplicity=1, ms=0.0),
        nroot=1,
    )(rhf)
    ci.run()
    assert ci.E[0] == approx(eci)

    mc = MCOptimizer()(ci)
    mc.run()
    assert mc.E == approx(ecasscf)
