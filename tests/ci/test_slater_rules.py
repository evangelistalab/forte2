import numpy as np

from forte2 import *
from forte2.jkbuilder import RestrictedMOIntegrals


def test_slater_rules_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="sto-6g", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-12)(system)
    scf.run()

    orbitals = [0, 1]
    norb = len(orbitals)
    ints = RestrictedMOIntegrals(system=scf.system, C=scf.C[0], orbitals=orbitals)

    slater_rules = forte2.SlaterRules(norb, ints.E, ints.H, ints.V)

    dets = forte2.hilbert_space(norb, scf.na, scf.nb)

    print(f"Determinants:")
    for i, det in enumerate(dets):
        print(f"{i}: {det}")

    H = np.zeros((len(dets), len(dets)))
    for i, I in enumerate(dets):
        for j, J in enumerate(dets):
            H[i, j] = slater_rules.slater_rules(I, J)

    print(f"Slater Rules Matrix:\n{H}")
    evals = np.linalg.eigvalsh(H)
    print(f"Eigenvalues: {evals}")

    assert np.isclose(
        evals[0], -1.096071975854
    ), "Slater rules test failed for H2 molecule"


def test_slater_rules_2():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-10)(system)
    scf.run()

    core_orbitals = [0]
    orbitals = [1, 2, 3, 4, 5, 6]  # Active orbitals
    norb = len(orbitals)
    ints = RestrictedMOIntegrals(
        system=scf.system, C=scf.C[0], orbitals=orbitals, core_orbitals=core_orbitals
    )

    slater_rules = forte2.SlaterRules(norb, ints.E, ints.H, ints.V)

    nca = scf.na - len(core_orbitals)
    ncb = scf.nb - len(core_orbitals)

    dets = forte2.hilbert_space(norb, nca, ncb)

    H = np.zeros((len(dets), len(dets)))
    for i, I in enumerate(dets):
        for j, J in enumerate(dets):
            H[i, j] = slater_rules.slater_rules(I, J)

    E = np.linalg.eigvalsh(H)[0]
    print(f"Lowest eigenvalue: {E}")

    assert np.isclose(
        E, -100.019788438077, atol=1e-9
    ), "Slater rules test failed for HF molecule"


if __name__ == "__main__":
    test_slater_rules_1()
    test_slater_rules_2()
