import numpy as np

import forte2
from forte2 import System, RHF
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.helpers.comparisons import approx
from forte2.ci.ci import _CISingleStateSolver
from forte2.state import MOSpace, State
from forte2.orbitals import convert_coeff_spatial_to_spinor
from forte2.base_classes import DavidsonLiuParams


def _determinant(alpha_occ, beta_occ):
    det = forte2.Determinant.zero()
    for p in alpha_occ:
        det.set_na(p, True)
    for p in beta_occ:
        det.set_nb(p, True)
    return det


def _symmetric_integrals(norb):
    rng = np.random.default_rng(12345)
    h = rng.normal(size=(norb, norb))
    h = 0.5 * (h + h.T)
    v = rng.normal(size=(norb, norb, norb, norb))
    v = 0.5 * (v + v.transpose(1, 0, 3, 2))
    v = 0.5 * (v + v.transpose(2, 3, 0, 1))
    return h, v


def _main_diagonal_energy(norb, scalar_energy, h, v, det):
    """Reference the main-branch diagonal Slater-rule loop structure."""

    alpha = [p for p in range(norb) if det.na(p)]
    beta = [p for p in range(norb) if det.nb(p)]

    energy = scalar_energy
    for a_idx, p in enumerate(alpha):
        energy += h[p, p]
        for q in alpha[a_idx + 1 :]:
            energy += v[p, q, p, q] - v[p, q, q, p]
        for q in beta:
            energy += v[p, q, p, q]

    for b_idx, p in enumerate(beta):
        energy += h[p, p]
        for q in beta[b_idx + 1 :]:
            energy += v[p, q, p, q] - v[p, q, q, p]

    return energy


def test_slater_rules_diagonal_edge_cases_match_main_formula():
    norb = 4
    scalar_energy = 0.37
    h, v = _symmetric_integrals(norb)
    slater_rules = forte2.SlaterRules(norb, scalar_energy, h, v)

    dets = [
        _determinant([], []),  # no electrons
        _determinant(range(norb), range(norb)),  # all active spin orbitals occupied
        _determinant([0, norb - 1], [1, norb - 2]),  # first/last active orbitals
    ]

    for det in dets:
        expected = _main_diagonal_energy(norb, scalar_energy, h, v, det)
        assert slater_rules.energy(det) == approx(expected)
        assert slater_rules.slater_rules(det, det) == approx(expected)
        assert slater_rules.slater_rules_reference(det, det) == approx(expected)


def test_slater_rules_returns_zero_for_incompatible_determinants():
    norb = 4
    h, v = _symmetric_integrals(norb)
    slater_rules = forte2.SlaterRules(norb, 0.0, h, v)

    cases = [
        (_determinant([0], []), _determinant([], [])),  # different electron count
        (_determinant([0], [0]), _determinant([], [])),  # different electron count
        (_determinant([0, 1], []), _determinant([], [])),  # different electron count
        (_determinant([0, 1], []), _determinant([0], [1])),  # same N, different Ms
        (_determinant([0, 1], [0]), _determinant([0], [0, 1])),  # same N, different Ms
        (_determinant([0, 1], [0, 1]), _determinant([2, 3], [2, 3])),  # rank > 2
    ]

    for lhs, rhs in cases:
        assert slater_rules.slater_rules(lhs, rhs) == 0.0
        assert slater_rules.slater_rules_reference(lhs, rhs) == 0.0


def test_slater_rules_fast_matches_reference_for_excitation_classes():
    norb = 8
    h, v = _symmetric_integrals(norb)
    slater_rules = forte2.SlaterRules(norb, 0.37, h, v)

    rhs = _determinant([0, 2, 5], [1, 3, 6])
    cases = [
        rhs,  # diagonal
        _determinant([0, 4, 5], [1, 3, 6]),  # alpha single
        _determinant([0, 2, 5], [1, 4, 6]),  # beta single
        _determinant([1, 4, 5], [1, 3, 6]),  # alpha-alpha double
        _determinant([0, 2, 5], [0, 4, 6]),  # beta-beta double
        _determinant([0, 4, 5], [1, 4, 6]),  # alpha-beta double
        _determinant([1, 3, 4], [1, 3, 6]),  # higher-rank alpha excitation
        _determinant([0, 2, 5, 7], [1, 3, 6]),  # unequal alpha electron count
    ]

    for lhs in cases:
        for left, right in ((lhs, rhs), (rhs, lhs)):
            fast = slater_rules.slater_rules(left, right)
            reference = slater_rules.slater_rules_reference(left, right)
            assert fast == approx(reference)


def test_slater_rules_fast_matches_reference_exhaustive_small_space():
    norb = 4
    h, v = _symmetric_integrals(norb)
    slater_rules = forte2.SlaterRules(norb, -0.19, h, v)

    dets = []
    for alpha_mask in range(1 << norb):
        alpha = [p for p in range(norb) if alpha_mask & (1 << p)]
        for beta_mask in range(1 << norb):
            beta = [p for p in range(norb) if beta_mask & (1 << p)]
            dets.append(_determinant(alpha, beta))

    for lhs in dets:
        for rhs in dets:
            fast = slater_rules.slater_rules(lhs, rhs)
            reference = slater_rules.slater_rules_reference(lhs, rhs)
            assert fast == approx(reference)


def test_slater_rules_1():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, e_tol=1e-12)(system)
    scf.run()

    orbitals = [0, 1]
    norb = len(orbitals)
    ints = RestrictedMOIntegrals(system=scf.system, C=scf.C[0], orbitals=orbitals)

    slater_rules = forte2.SlaterRules(norb, ints.E, ints.H, ints.V)

    dets = forte2.hilbert_space(norb, scf.na, scf.nb)

    H = np.zeros((len(dets), len(dets)))
    H_reference = np.zeros_like(H)
    for i, I in enumerate(dets):
        for j, J in enumerate(dets):
            H[i, j] = slater_rules.slater_rules(I, J)
            H_reference[i, j] = slater_rules.slater_rules_reference(I, J)

    assert np.allclose(H, H_reference, atol=1e-12)

    evals = np.linalg.eigvalsh(H)
    evals_reference = np.linalg.eigvalsh(H_reference)

    assert np.isclose(
        evals[0], -1.096071975854
    ), "Slater rules test failed for H2 molecule"
    assert np.isclose(
        evals_reference[0], -1.096071975854
    ), "Reference Slater rules test failed for H2 molecule"


def test_slater_rules_2():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, e_tol=1e-10)(system)
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
    scf = RHF(charge=0, e_tol=1e-12)(system)
    scf.run()

    C = convert_coeff_spatial_to_spinor(scf.C)
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
    scf = RHF(charge=0, e_tol=1e-10)(system)
    scf.run()

    C = convert_coeff_spatial_to_spinor(scf.C)
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
        norb, ints.E.real, ints.H.astype(complex), ints.V.astype(complex)
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
    ci = _CISingleStateSolver(
        mo_space=mo_space,
        state=state,
        ints=fakeints,
        nroot=1,
        active_orbsym=[[0] * norb],
        do_test_rdms=True,
        two_component=True,
        davidson_liu_params=DavidsonLiuParams(maxiter=200),
    )
    ci.run()

    assert E == approx(eref)
    assert E == approx(ci.E[0])


def test_slater_rules_4_complex_antisym():
    # same setup as above, but use antisymmetrized TEIs.

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
    h2 -= h2.swapaxes(2, 3)

    slater_rules = forte2.RelSlaterRules(norb, 0.0, h1, h2, tei_is_asym=True)
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
    ci = _CISingleStateSolver(
        mo_space=mo_space,
        state=state,
        ints=fakeints,
        nroot=1,
        active_orbsym=[[0] * norb],
        davidson_liu_params=DavidsonLiuParams(maxiter=200),
        do_test_rdms=True,
        two_component=True,
    )
    ci.run(use_asym_ints=True)

    assert E == approx(eref)
    assert E == approx(ci.E[0])
