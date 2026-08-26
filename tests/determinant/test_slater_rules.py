import numpy as np
import pytest

from forte2 import System, RHF
from forte2.lib.det import Determinant, SlaterRules, RelSlaterRules, hilbert_space
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.helpers.comparisons import approx
from forte2.ci.rel_ci import _RelCISingleStateSolver
from forte2.state import MOSpace, State
from forte2.base_classes import DavidsonLiuParams


def _determinant(alpha_occ, beta_occ):
    """Helper function to construct a Determinant from lists of occupied alpha and beta orbitals."""
    det = Determinant.zero()
    for p in alpha_occ:
        det.set_na(p, True)
    for p in beta_occ:
        det.set_nb(p, True)
    return det


def _symmetric_integrals(norb, seed=12345):
    """Helper function to generate random symmetric one- and two-electron integrals.
    The final integrals have the following symmetries:
    h[p, q] = h[q, p]
    v[p, q, r, s] = v[q, p, s, r] = v[r, s, p, q] = v[s, r, q, p]
    """
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(norb, norb))
    h = 0.5 * (h + h.T)
    v = rng.normal(size=(norb, norb, norb, norb))
    v = 0.5 * (v + v.transpose(1, 0, 3, 2))
    v = 0.5 * (v + v.transpose(2, 3, 0, 1))
    # verify symmetries before returning
    assert np.allclose(h, h.T), "One-electron integrals are not symmetric"
    assert np.allclose(
        v, v.transpose(1, 0, 3, 2)
    ), "Two-electron integrals do not have p<->q, r<->s symmetry"
    assert np.allclose(
        v, v.transpose(2, 3, 0, 1)
    ), "Two-electron integrals do not have (pq)<->(rs) symmetry"
    assert np.allclose(
        v, v.transpose(3, 2, 1, 0)
    ), "Two-electron integrals do not have p<->s, q<->r symmetry"
    return h, v


def _random_determinants(norb, nalpha, nbeta, ndets, seed=67890):
    """Helper function to generate a list of random determinants with specified numbers of alpha and beta electrons."""
    rng = np.random.default_rng(seed)
    dets = []
    seen = set()
    while len(dets) < ndets:
        alpha = tuple(sorted(rng.choice(norb, size=nalpha, replace=False)))
        beta = tuple(sorted(rng.choice(norb, size=nbeta, replace=False)))
        key = (alpha, beta)
        if key in seen:
            continue
        seen.add(key)
        dets.append(_determinant(alpha, beta))
    return dets


def _reference_determinant_energy(norb, scalar_energy, h, v, det):
    """Reference implementation of the energy of a determinant."""

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


def _reference_slater_rule(norb, scalar_energy, h, v, det_i, det_j):
    """Reference implementation of the Slater rules between two determinants."""

    # Get the sets of occupied alpha and beta orbitals for each determinant
    i_alpha = {p for p in range(norb) if det_i.na(p)}
    i_beta = {p for p in range(norb) if det_i.nb(p)}
    j_alpha = {p for p in range(norb) if det_j.na(p)}
    j_beta = {p for p in range(norb) if det_j.nb(p)}

    # check that the number of electrons is the same, otherwise the matrix element is zero
    if len(i_alpha) != len(j_alpha) or len(i_beta) != len(j_beta):
        return 0.0

    # Compute the differences in occupied orbitals between the two determinants
    i_alpha_diff = list(i_alpha - j_alpha)
    j_alpha_diff = list(j_alpha - i_alpha)
    i_beta_diff = list(i_beta - j_beta)
    j_beta_diff = list(j_beta - i_beta)

    # Count the number of different orbitals in the alpha and beta strings
    n_diff_alpha = len(i_alpha_diff) + len(j_alpha_diff)
    n_diff_beta = len(i_beta_diff) + len(j_beta_diff)

    if n_diff_alpha + n_diff_beta == 0:
        return _reference_determinant_energy(norb, scalar_energy, h, v, det_i)
    elif n_diff_alpha == 2 and n_diff_beta == 0:
        p = i_alpha_diff[0]
        q = j_alpha_diff[0]
        new_det = Determinant(det_j)
        sign = new_det.destroy_alpha(q)
        sign *= new_det.create_alpha(p)
        energy = h[p, q]
        for r in i_alpha.intersection(j_alpha):
            energy += v[p, r, q, r] - v[p, r, r, q]
        for r in i_beta.intersection(j_beta):
            energy += v[p, r, q, r]
        return sign * energy
    elif n_diff_alpha == 0 and n_diff_beta == 2:
        p = i_beta_diff[0]
        q = j_beta_diff[0]
        new_det = Determinant(det_j)
        sign = new_det.destroy_beta(q)
        sign *= new_det.create_beta(p)
        energy = h[p, q]
        for r in i_beta.intersection(j_beta):
            energy += v[p, r, q, r] - v[p, r, r, q]
        for r in i_alpha.intersection(j_alpha):
            energy += v[r, p, r, q]
        return sign * energy
    elif n_diff_alpha == 4 and n_diff_beta == 0:
        p, q = i_alpha_diff
        r, s = j_alpha_diff
        new_det = Determinant(det_j)
        sign = new_det.destroy_alpha(r)
        sign *= new_det.destroy_alpha(s)
        sign *= new_det.create_alpha(q)
        sign *= new_det.create_alpha(p)
        return sign * (v[p, q, r, s] - v[p, q, s, r])
    elif n_diff_alpha == 0 and n_diff_beta == 4:
        p, q = i_beta_diff
        r, s = j_beta_diff
        new_det = Determinant(det_j)
        sign = new_det.destroy_beta(r)
        sign *= new_det.destroy_beta(s)
        sign *= new_det.create_beta(q)
        sign *= new_det.create_beta(p)
        return sign * (v[p, q, r, s] - v[p, q, s, r])
    elif n_diff_alpha == 2 and n_diff_beta == 2:
        p = i_alpha_diff[0]
        q = i_beta_diff[0]
        r = j_alpha_diff[0]
        s = j_beta_diff[0]
        new_det = Determinant(det_j)
        sign = new_det.destroy_alpha(r)
        sign *= new_det.destroy_beta(s)
        sign *= new_det.create_beta(q)
        sign *= new_det.create_alpha(p)
        return sign * v[p, q, r, s]
    else:
        return 0.0


def test_slater_rules_rejects_negative_norb():
    """Test that SlaterRules raises a ValueError if norb is negative."""
    h = np.zeros((0, 0))
    v = np.zeros((0, 0, 0, 0))

    with pytest.raises(
        ValueError, match="SlaterRules: norb must be non-negative, got -1"
    ):
        SlaterRules(-1, 0.0, h, v)


def _random_hermitian_integrals(nspinor, seed):
    """Helper function to generate random Hermitian one- and antisymmetrizable two-electron
    integrals suitable for RelSlaterRules."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(nspinor, nspinor)) + 1j * rng.normal(size=(nspinor, nspinor))
    h = h + h.T.conj()
    v = rng.normal(size=(nspinor,) * 4) + 1j * rng.normal(size=(nspinor,) * 4)
    v = v + v.transpose(1, 0, 3, 2)  # electron exchange: <pq|rs> = <qp|sr>
    v = v + v.transpose(2, 3, 0, 1).conj()  # Hermiticity: <pq|rs> = <rs|pq>*
    return h, v


@pytest.mark.parametrize("which", ["E", "H", "V"])
def test_rel_slater_rules_update_integrals_partial_update_keeps_other_arguments(which):
    """update_integrals with only one of E/H/V given must update only that argument, matching
    a fresh RelSlaterRules built with the same effective (E, H, V)."""
    nspinor = 6
    h_a, v_a = _random_hermitian_integrals(nspinor, seed=1)
    h_b, v_b = _random_hermitian_integrals(nspinor, seed=2)
    e_a, e_b = 0.1, 0.2

    reused = RelSlaterRules(nspinor, e_a, h_a, v_a)
    if which == "E":
        reused.update_integrals(nspinor, scalar_energy=e_b)
    elif which == "H":
        reused.update_integrals(nspinor, one_electron_integrals=h_b)
    else:
        reused.update_integrals(nspinor, two_electron_integrals=v_b)

    fresh = RelSlaterRules(
        nspinor,
        e_b if which == "E" else e_a,
        h_b if which == "H" else h_a,
        v_b if which == "V" else v_a,
    )

    dets = _random_determinants(nspinor, 3, 0, ndets=5)
    for det in dets:
        assert reused.energy(det) == approx(fresh.energy(det))


def test_rel_slater_rules_update_integrals_nspinor_change_requires_h_and_v_together():
    """RelSlaterRules's owned H/V copies are sized by nspinor, so a stale one left over from a
    different nspinor would be read out of bounds; giving only one of H/V while nspinor changes
    must raise, and giving both together must succeed."""
    nspinor_a, nspinor_b = 4, 6
    h_a, v_a = _random_hermitian_integrals(nspinor_a, seed=1)
    h_b, v_b = _random_hermitian_integrals(nspinor_b, seed=2)

    rsr = RelSlaterRules(nspinor_a, 0.1, h_a, v_a)
    with pytest.raises(RuntimeError, match="given together"):
        rsr.update_integrals(nspinor_b, one_electron_integrals=h_b)
    with pytest.raises(RuntimeError, match="given together"):
        rsr.update_integrals(nspinor_b, two_electron_integrals=v_b)

    rsr.update_integrals(nspinor_b, 0.2, h_b, v_b)
    fresh = RelSlaterRules(nspinor_b, 0.2, h_b, v_b)
    for det in _random_determinants(nspinor_b, 3, 0, ndets=5):
        assert rsr.energy(det) == approx(fresh.energy(det))


def test_rel_slater_rules_copies_integrals_not_aliases():
    """RelSlaterRules must deep-copy H/V at construction/update_integrals time rather than
    alias the numpy arrays: mutating H/V afterward from Python must not change subsequently
    computed energies."""
    nspinor = 5
    h, v = _random_hermitian_integrals(nspinor, seed=3)
    rsr = RelSlaterRules(nspinor, 0.1, h, v)

    dets = _random_determinants(nspinor, 3, 0, ndets=5)
    energies_before = [rsr.energy(det) for det in dets]

    h[:] = 0.0
    v[:] = 0.0

    energies_after = [rsr.energy(det) for det in dets]
    assert energies_after == approx(energies_before)


def test_slater_rules_diagonal_edge_cases_match_main_formula():
    """Test that the energy of edge case determinants matches the main diagonal formula."""
    norb = 4
    scalar_energy = 0.37
    h, v = _symmetric_integrals(norb)
    slater_rules = SlaterRules(norb, scalar_energy, h, v)

    dets = [
        _determinant([], []),  # no electrons
        _determinant(range(norb), range(norb)),  # all active spin orbitals occupied
        _determinant([0, norb - 1], [1, norb - 2]),  # first/last active orbitals
    ]

    for det in dets:
        expected = _reference_determinant_energy(norb, scalar_energy, h, v, det)
        assert slater_rules.energy(det) == approx(expected)
        assert slater_rules.slater_rules(det, det) == approx(expected)


def test_slater_rules_returns_zero_for_incompatible_determinants():
    """Test that SlaterRules returns zero for determinants that differ by more than 2 spin orbitals and different number of electrons."""
    norb = 4
    h, v = _symmetric_integrals(norb)
    slater_rules = SlaterRules(norb, 0.0, h, v)

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


def test_slater_rules_matches_reference_for_excitation_classes():
    """Test that SlaterRules matches the reference implementation for specific pairs of determinants representing different excitation classes."""
    norb = 8
    h, v = _symmetric_integrals(norb)
    slater_rules = SlaterRules(norb, 0.37, h, v)

    rhs = _determinant([0, 2, 5], [1, 3, 6])
    cases = [
        rhs,  # diagonal
        _determinant([0, 4, 5], [1, 3, 6]),  # alpha single
        _determinant([0, 2, 5], [1, 4, 6]),  # beta single
        _determinant([1, 4, 5], [1, 3, 6]),  # alpha-alpha double
        _determinant([0, 2, 5], [0, 4, 6]),  # beta-beta double
        _determinant([0, 4, 5], [1, 4, 6]),  # alpha-beta double
        _determinant([1, 3, 5], [1, 3, 6]),  # connected alpha-alpha double
        _determinant([1, 3, 4], [1, 3, 6]),  # disconnected rank-3 alpha excitation
        _determinant([0, 2, 5, 7], [1, 3, 6]),  # unequal alpha electron count
    ]

    for lhs in cases:
        print(f"Testing Slater rules between:\n{lhs}\n and\n{rhs}")
        expected = _reference_slater_rule(norb, 0.37, h, v, lhs, rhs)
        found = slater_rules.slater_rules(lhs, rhs)
        print(f"Expected value: {expected}")
        print(f"Found value:    {found}")
        assert found == approx(expected)


def test_slater_rules_matches_reference_for_fock_space():
    """Test that SlaterRules matches the reference implementation for all pairs of determinants in the Fock space."""
    norb = 4
    h, v = _symmetric_integrals(norb)
    slater_rules = SlaterRules(norb, 0.37, h, v)

    fock_space_dets = []
    for nalpha in range(norb + 1):
        for nbeta in range(norb + 1):
            dets = hilbert_space(norb, nalpha, nbeta)
            fock_space_dets.extend(dets)

    for lhs in fock_space_dets:
        for rhs in fock_space_dets:
            expected = _reference_slater_rule(norb, 0.37, h, v, lhs, rhs)
            found = slater_rules.slater_rules(lhs, rhs)
            assert found == approx(expected)


def test_slater_rules_1():
    """Test SlaterRules by building the FCI matrix and finding the ground state energy."""
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
        system=scf.system,
        C=scf.mos.C[0],
        orbitals=orbitals,
        core_orbitals=core_orbitals,
    )

    slater_rules = SlaterRules(norb, ints.E, ints.H, ints.V)

    nca = scf.na - len(core_orbitals)
    ncb = scf.nb - len(core_orbitals)

    dets = hilbert_space(norb, nca, ncb)

    H = np.zeros((len(dets), len(dets)))
    for i, I in enumerate(dets):
        for j, J in enumerate(dets):
            H[i, j] = slater_rules.slater_rules(I, J)

    E = np.linalg.eigvalsh(H)[0]

    assert np.isclose(
        E, -100.019788438077, atol=1e-9
    ), "Slater rules test failed for HF molecule"


def test_slater_rules_1_complex():
    """Test SlaterRules with complex integrals by building the FCI matrix and finding the ground state energy."""
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, e_tol=1e-12)(system)
    scf.run()

    mos_2c = scf.mos.to_spinorbital_basis()
    orbitals = [0, 1, 2, 3]
    norb = len(orbitals)
    system.two_component = True
    ints = SpinorbitalIntegrals(system=system, C=mos_2c.C[0], spinorbitals=orbitals)

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

    slater_rules = RelSlaterRules(
        norb, ints.E, ints.H.astype(complex), ints.V.astype(complex)
    )

    dets = hilbert_space(norb, scf.na + scf.nb, 0)

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
    """Test SlaterRules with complex integrals by building the FCI matrix and finding the ground state energy."""
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, e_tol=1e-10)(system)
    scf.run()

    mos_2c = scf.mos.to_spinorbital_basis()
    system.two_component = True

    core_orbitals = [0, 1]
    orbitals = list(range(2, 14))  # Active orbitals
    norb = len(orbitals)
    ints = SpinorbitalIntegrals(
        system=scf.system,
        C=mos_2c.C[0],
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

    slater_rules = RelSlaterRules(
        norb, ints.E.real, ints.H.astype(complex), ints.V.astype(complex)
    )

    nca = scf.na - len(core_orbitals) // 2
    ncb = scf.nb - len(core_orbitals) // 2

    dets = hilbert_space(norb, nca + ncb, 0)

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
    """Test SlaterRules with complex integrals by building the FCI matrix and finding the ground state energy."""
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

    slater_rules = RelSlaterRules(norb, 0.0, h1, h2)
    dets = hilbert_space(norb, 8, 0)
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
    ci = _RelCISingleStateSolver(
        mo_space=mo_space,
        state=state,
        ints=fakeints,
        nroot=1,
        active_orbsym=[[0] * norb],
        do_test_rdms=True,
        davidson_liu_params=DavidsonLiuParams(maxiter=200),
    )
    ci.run()

    assert E == approx(eref)
    assert E == approx(ci.E[0])


def test_slater_rules_4_complex_antisym():
    """Test SlaterRules with complex antisymmetrized integrals by building the FCI matrix and finding the ground state energy."""
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

    # h2 is already antisymmetric (<pq||rs>). RelSlaterRules and the CI solver
    # antisymmetrize it, doubling it, hence the factor of 0.5
    slater_rules = RelSlaterRules(norb, 0.0, h1, 0.5 * h2)
    dets = hilbert_space(norb, 8, 0)
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
    fakeints.V = 0.5 * h2
    mo_space = MOSpace(nmo=norb, active_orbitals=list(range(norb)))
    state = State(nel=8, multiplicity=1, ms=0.0)
    ci = _RelCISingleStateSolver(
        mo_space=mo_space,
        state=state,
        ints=fakeints,
        nroot=1,
        active_orbsym=[[0] * norb],
        davidson_liu_params=DavidsonLiuParams(maxiter=200),
        do_test_rdms=True,
    )
    ci.run()

    assert E == approx(eref)
    assert E == approx(ci.E[0])
