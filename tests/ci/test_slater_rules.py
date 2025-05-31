from numpy import isclose

from forte2 import *


def test_slater_rules_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="sto-6g", auxiliary_basis="cc-pVTZ-JKFIT", units="bohr"
    )
    scf = RHF(charge=0, econv=1e-12)(system)
    scf.run()

    orbitals = [0, 1]
    norb = len(orbitals)
    ints = RestrictedMOIntegrals(system=scf.system, C=scf.C[0], orbitals=orbitals)

    slater_rules = SlaterRules(norb, ints.E, ints.H, ints.V)

    dets = forte2.hilbert_space(norb, scf.na, scf.nb)

    H = np.zeros((len(dets), len(dets)))
    for i, I in enumerate(dets):
        for j, J in enumerate(dets):
            H[i, j] = slater_rules.slater_rule(I, J)

    print("Slater Rules Matrix:")
    print(H)
    E = np.linalg.eigvalsh(H)[0]
    print(f"Lowest eigenvalue: {E}")


if __name__ == "__main__":
    test_slater_rules_1()
