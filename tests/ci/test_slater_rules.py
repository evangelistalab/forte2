import numpy as np

import forte2
from forte2 import System, RHF
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.helpers.comparisons import approx
from forte2.ci.ci import _CIBase
from forte2.state import MOSpace, State
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor


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

    H = np.zeros((len(dets), len(dets)))
    for i, I in enumerate(dets):
        for j, J in enumerate(dets):
            H[i, j] = slater_rules.slater_rules(I, J)

    evals = np.linalg.eigvalsh(H)

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

    assert np.isclose(
        E, -100.019788438077, atol=1e-9
    ), "Slater rules test failed for HF molecule"


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

    C = convert_coeff_spatial_to_spinor(system, scf.C)
    orbitals = [0, 1, 2, 3]
    norb = len(orbitals)
    system.two_component = True
    ints = SpinorbitalIntegrals(system=system, C=C[0], spinorbitals=orbitals)

    random_phase = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, size=norb)))
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

    dets = forte2.hilbert_space(norb, scf.na + scf.nb, 0)

    H = np.zeros((len(dets), len(dets)), dtype=complex)
    for i in range(len(dets)):
        # no triangular loop: explicitly construct both i,j and j,i to check Hermiticity
        for j in range(len(dets)):
            H[i, j] = slater_rules.slater_rules(dets[i], dets[j])

    assert np.allclose(H, H.T.conj()), "Slater rules matrix is not Hermitian"

    evals = np.linalg.eigvalsh(H)

    assert np.isclose(
        evals[0], -1.096071975854
    ), "Slater rules test failed for H2 molecule"


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

    C = convert_coeff_spatial_to_spinor(system, scf.C)
    system.two_component = True

    core_orbitals = [0, 1]
    orbitals = list(range(2, 14))  # Active orbitals
    norb = len(orbitals)
    ints = SpinorbitalIntegrals(
        system=scf.system,
        C=C[0],
        spinorbitals=orbitals,
        core_spinorbitals=core_orbitals,
    )

    random_phase = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, size=norb)))
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
        norb, ints.E, ints.H.astype(complex), ints.V.astype(complex)
    )

    nca = scf.na - len(core_orbitals) // 2
    ncb = scf.nb - len(core_orbitals) // 2

    dets = forte2.hilbert_space(norb, nca + ncb, 0)

    H = np.zeros((len(dets), len(dets)), dtype=complex)
    for i in range(len(dets)):
        # no triangular loop: explicitly construct both i,j and j,i to check Hermiticity
        for j in range(len(dets)):
            H[i, j] = slater_rules.slater_rules(dets[i], dets[j])

    assert np.allclose(H, H.T.conj()), "Slater rules matrix is not Hermitian"

    E = np.linalg.eigvalsh(H)[0]

    assert np.isclose(
        E, -100.019788438077, atol=1e-9
    ), "Slater rules test failed for HF molecule"


def test_slater_rules_3_complex():
    # reference energy from pyscf:
    # np.random.seed(12)
    # norb = 12
    # h1 = np.random.random((norb,norb)) + 1j * np.random.random((norb,norb))
    # h2 = np.random.random((norb,norb,norb,norb)) + 1j * np.random.random((norb,norb,norb,norb))
    # print(np.linalg.norm(h1), np.linalg.norm(h2))
    # # Restore permutation symmetry
    # h1 = h1 + h1.T.conj()
    # h2 = h2 + h2.transpose(2, 3, 0, 1)
    # h2 = h2 + h2.transpose(1, 0, 3, 2).conj()
    # h2 = h2 + h2.transpose(3, 2, 1, 0).conj()
    # cisolver = pyscf.fci.fci_dhf_slow.FCI()
    # cisolver.max_cycle = 100
    # cisolver.conv_tol = 1e-12
    # e, fcivec = cisolver.kernel(h1, h2, norb, nelec=8, verbose=5)

    eref = -80.43551643145948
    rng = np.random.default_rng(12)
    norb = 12
    h1 = rng.random((norb, norb)) + 1j * rng.random((norb, norb))
    h2 = rng.random((norb, norb, norb, norb)) + 1j * rng.random(
        (norb, norb, norb, norb)
    )
    # Restore permutation symmetry
    h1 = h1 + h1.T.conj()
    # pyscf uses chemist's notation, forte2 uses physicist's notation
    h2 = h2.swapaxes(1, 2)
    h2 = h2 + h2.transpose(2, 3, 0, 1).conj()
    h2 = h2 + h2.transpose(1, 0, 3, 2)
    h2 = h2 + h2.transpose(3, 2, 1, 0).conj()

    slater_rules = forte2.RelSlaterRules(norb, 0.0, h1, h2)
    dets = forte2.hilbert_space(norb, 8, 0)
    H = np.zeros((len(dets), len(dets)), dtype=complex)
    for i in range(len(dets)):
        # no triangular loop: explicitly construct both i,j and j,i to check Hermiticity
        for j in range(len(dets)):
            H[i, j] = slater_rules.slater_rules(dets[i], dets[j])
    assert np.allclose(H, H.T.conj()), "Slater rules matrix is not Hermitian"
    E = np.linalg.eigvalsh(H)[0]

    fakeints = SpinorbitalIntegrals.__new__(SpinorbitalIntegrals)
    fakeints.E = 0.0
    fakeints.H = h1
    fakeints.V = h2
    mo_space = MOSpace(nmo=norb, active_orbitals=list(range(norb)))
    state = State(nel=8, multiplicity=1, ms=0.0)
    ci = _CIBase(
        mo_space=mo_space,
        state=state,
        ints=fakeints,
        nroot=1,
        active_orbsym=[[0] * norb],
        maxiter=200,
        do_test_rdms=True,
        ci_algorithm="hz",
        two_component=True,
    )
    ci.run()

    assert E == approx(eref)
    assert E == approx(ci.E[0])
