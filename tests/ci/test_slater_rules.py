import numpy as np

import forte2
from forte2 import System, RHF
from forte2.jkbuilder import RestrictedMOIntegrals


def test_slater_rules_1():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-12)(system)
    scf.run()

    orbitals = [0, 1]
    norb = len(orbitals)
    ints = RestrictedMOIntegrals(system=scf.system, C=scf.C[0], orbitals=orbitals)

    slater_rules = forte2.SlaterRules(norb, ints.E, ints.H, ints.V)

    dets = forte2.hilbert_space(norb, scf.na, scf.nb)

    print("Determinants:")
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


def test_slater_rules_1_complex():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-12)(system)
    scf.run()

    orbitals = [0, 1]
    norb = len(orbitals)
    ints = RestrictedMOIntegrals(
        system=scf.system, C=scf.C[0], orbitals=orbitals, spinorbital=True
    )

    random_phase = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, size=norb * 2)))
    ints.H = random_phase.T.conj() @ ints.H @ random_phase
    ints.V = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        ints.V,
        random_phase.conj(),
        random_phase.conj(),
        random_phase,
        random_phase,
        optimize=True,
    )

    slater_rules = forte2.RelSlaterRules(
        norb * 2, ints.E, ints.H.astype(complex), ints.V.astype(complex)
    )

    dets = forte2.hilbert_space(norb * 2, scf.na + scf.nb, 0)

    print("Determinants:")
    for i, det in enumerate(dets):
        print(f"{i}: {det}")
        print(f"energy: {slater_rules.energy(det)}")

    H = np.zeros((len(dets), len(dets)), dtype=complex)
    for i in range(len(dets)):
        for j in range(len(dets)):
            H[i, j] = slater_rules.slater_rules(dets[i], dets[j])

    assert np.allclose(H, H.T.conj()), "Slater rules matrix is not Hermitian"

    print(f"Slater Rules Matrix:\n{H}")
    evals = np.linalg.eigvalsh(H)
    print(f"Eigenvalues: {evals}")

    assert np.isclose(
        evals[0], -1.096071975854
    ), "Slater rules test failed for H2 molecule"


def test_slater_rules_2():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
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


def test_slater_rules_2_complex():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-10)(system)
    scf.run()

    core_orbitals = [0]
    orbitals = [1, 2, 3, 4, 5, 6]  # Active orbitals
    norb = len(orbitals)
    ints = RestrictedMOIntegrals(
        system=scf.system,
        C=scf.C[0],
        orbitals=orbitals,
        core_orbitals=core_orbitals,
        spinorbital=True,
    )

    random_phase = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, size=norb * 2)))
    ints.H = random_phase.T.conj() @ ints.H @ random_phase
    ints.V = np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        ints.V,
        random_phase.conj(),
        random_phase.conj(),
        random_phase,
        random_phase,
        optimize=True,
    )

    slater_rules = forte2.RelSlaterRules(
        norb * 2, ints.E, ints.H.astype(complex), ints.V.astype(complex)
    )

    nca = scf.na - len(core_orbitals)
    ncb = scf.nb - len(core_orbitals)

    dets = forte2.hilbert_space(norb * 2, nca + ncb, 0)

    H = np.zeros((len(dets), len(dets)), dtype=complex)
    for i in range(len(dets)):
        for j in range(len(dets)):
            H[i, j] = slater_rules.slater_rules(dets[i], dets[j])

    assert np.allclose(H, H.T.conj()), "Slater rules matrix is not Hermitian"

    E = np.linalg.eigvalsh(H)[0]
    print(f"Lowest eigenvalue: {E}")

    assert np.isclose(
        E, -100.019788438077, atol=1e-9
    ), "Slater rules test failed for HF molecule"
