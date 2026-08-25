import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.linalg import expm, logm

from forte2 import System, RHF, CI, CISolver, MCOptimizer, State, GHF, RelCI
from forte2.helpers.comparisons import approx_abs
from forte2.orbitals.orbital_overlap import mo_overlap
from forte2.orbitals.wavefunction_overlap import (
    biorthogonalize_casscf_orbitals,
    casscf_wavefunction_overlap,
    transform_ci_vector_direct,
    transform_ci_vector_sparse_ops,
    _one_body_sparse_operator,
    _make_sparse_state,
)

BACKENDS = ["direct", "sparse_ops"]


def random_orthonormal(nbf, n, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((nbf, n)))
    return Q


@dataclass
class CIState:
    """Everything casscf_wavefunction_overlap needs from one converged solver."""

    ci_strings: object
    C: np.ndarray
    C_docc_actv: np.ndarray
    ndocc: int
    nactv: int
    solver: object

    def overlap_args(self):
        """The (ci_strings, C, C_docc_actv) triple, in dispatcher order."""
        return (self.ci_strings, self.C, self.C_docc_actv)


def _ci_state(ci, root=0):
    """
    Extract the determinant-basis CI vector and docc+active MO coefficients
    from a converged CI/RelCI/MCOptimizer object.

    Two-component solvers already work in the determinant basis, whereas
    nonrelativistic ones solve in the CSF basis and must be converted; that
    asymmetry is handled here rather than at every call site.
    """
    solver = (
        ci.ci_solver.sub_solvers[0] if hasattr(ci, "ci_solver") else ci.sub_solvers[0]
    )
    space = ci.mo_space
    n = space.ncore + space.nactv
    if hasattr(solver, "spin_adapter"):
        C = np.zeros(solver.ndet)
        solver.spin_adapter.csf_C_to_det_C(solver.evecs[:, root], C)
    else:
        C = solver.evecs[:, root].copy()
    return CIState(
        ci_strings=solver.ci_strings,
        C=C,
        C_docc_actv=ci.mos.C[0][:, space.orig_to_contig][:, :n],
        ndocc=space.ncore,
        nactv=space.nactv,
        solver=solver,
    )


def _sparse_to_dense(state, dets):
    """
    Densify a SparseState onto a determinant list, in that list's order.
    Always complex, since SparseState coefficients are; real callers take
    ``.real`` of the result.
    """
    return np.array([state[d] if d in state else 0.0 for d in dets], dtype=complex)


@pytest.mark.parametrize(
    "nbf,ndocc,nactv",
    [(30, 3, 4), (20, 0, 5), (20, 5, 1)],
)
def test_biorthogonalize_casscf_orbitals(nbf, ndocc, nactv):
    n = ndocc + nactv
    rng = np.random.default_rng(0)
    C_X = random_orthonormal(nbf, n, seed=1)
    C_Y = random_orthonormal(nbf, n, seed=2)
    S_AO = rng.standard_normal((nbf, nbf))
    S_AO = S_AO @ S_AO.T + nbf * np.eye(nbf)
    S_XY = C_X.T @ S_AO @ C_Y

    bio = biorthogonalize_casscf_orbitals(S_XY, ndocc, nactv)

    S_AB = bio.C_XA.conj().T @ S_XY @ bio.C_YB
    assert np.max(np.abs(S_AB - np.eye(n))) == approx_abs(0.0, 1e-10)

    # the transformation must be block upper-triangular: new docc orbitals
    # are pure recombinations of old docc orbitals only.
    leak = bio.C_XA[ndocc:, :ndocc]
    if leak.size:
        assert np.max(np.abs(leak)) == approx_abs(0.0, 1e-10)

    # the returned active-space factors must reconstruct the assembled active
    # blocks exactly -- this is what lets the direct backend apply the
    # orthogonal rotation and the diagonal rescale as separate exact steps
    # instead of taking a matrix logarithm of their product.
    assert bio.U_actv_A == approx_abs(bio.C_XA[ndocc:, ndocc:], 1e-13)
    assert bio.U_actv_B @ np.diag(1.0 / bio.d_actv) == approx_abs(
        bio.C_YB[ndocc:, ndocc:], 1e-13
    )
    assert bio.U_actv_A.T @ bio.U_actv_A == approx_abs(np.eye(nactv), 1e-12)
    assert bio.U_actv_B.T @ bio.U_actv_B == approx_abs(np.eye(nactv), 1e-12)
    assert np.all(bio.d_actv > 0)


def _lih_ci():
    xyz = "Li 0.0 0.0 0.0\nH  0.0 0.0 3.0"
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4],
    )(rhf)
    ci.run()
    return system, ci


@pytest.mark.parametrize("backend", BACKENDS)
def test_casscf_wavefunction_overlap_same_orbitals(backend):
    system, ci = _lih_ci()
    st = _ci_state(ci)

    S = casscf_wavefunction_overlap(
        *st.overlap_args(),
        system,
        *st.overlap_args(),
        system,
        st.ndocc,
        st.nactv,
        backend=backend,
    )
    assert S == approx_abs(1.0, 1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_casscf_wavefunction_overlap_rotated_active_orbitals(backend):
    """
    Rotate the active orbitals by a random unitary R and independently build
    the CI vector of the SAME physical wavefunction in the rotated basis
    (via SparseExp/apply_op on the antihermitian generator log(R), the
    existing sparse-operator infrastructure). The biorthogonalized overlap
    between the original and the rotated representation must recover 1.
    """
    from forte2.lib.sparse_ops import SparseExp

    system, ci = _lih_ci()
    st = _ci_state(ci)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((st.nactv, st.nactv))
    R = expm(A - A.T)

    state1 = _make_sparse_state(st.ci_strings, st.C, 1e-14)
    T_op = _one_body_sparse_operator(logm(R))
    state2_ref = SparseExp(32, 1e-14).apply_op(T_op, state1, scaling_factor=-1.0)

    dets = st.ci_strings.make_determinants()
    C2_det = _sparse_to_dense(state2_ref, dets).real

    C_docc_actv_2 = st.C_docc_actv.copy()
    C_docc_actv_2[:, st.ndocc :] = st.C_docc_actv[:, st.ndocc :] @ R

    S = casscf_wavefunction_overlap(
        *st.overlap_args(),
        system,
        st.ci_strings,
        C2_det,
        C_docc_actv_2,
        system,
        st.ndocc,
        st.nactv,
        backend=backend,
    )
    assert S == approx_abs(1.0, 1e-8)


def _brute_force_determinant_overlap(S_XY, ndocc, dets1, C1, dets2, C2, screen=1e-13):
    """
    O(ndet1 * ndet2) Loewdin determinant-overlap calculation, ``<Psi1|Psi2>``,
    completely independent of the biorthogonalization/sparse-ops machinery in
    ``wavefunction_overlap.py`` (no orbital transformation, just a direct sum
    of Slater-determinant overlaps via the determinant-of-mutual-spin-orbital-
    overlaps formula). Intended only as a cross-check on small test systems;
    scales as the product of the two determinant counts.
    """
    nactv = S_XY.shape[0] - ndocc
    docc_idx = list(range(ndocc))
    total = 0.0 + 0.0j
    for k, d1 in enumerate(dets1):
        c1 = C1[k]
        if abs(c1) < screen:
            continue
        occ_a1 = docc_idx + [ndocc + p for p in range(nactv) if d1.na(p)]
        occ_b1 = docc_idx + [ndocc + p for p in range(nactv) if d1.nb(p)]
        for l, d2 in enumerate(dets2):
            c2 = C2[l]
            if abs(c2) < screen:
                continue
            occ_a2 = docc_idx + [ndocc + p for p in range(nactv) if d2.na(p)]
            occ_b2 = docc_idx + [ndocc + p for p in range(nactv) if d2.nb(p)]
            det_a = np.linalg.det(S_XY[np.ix_(occ_a1, occ_a2)]) if occ_a1 else 1.0
            det_b = np.linalg.det(S_XY[np.ix_(occ_b1, occ_b2)]) if occ_b1 else 1.0
            total += np.conj(c1) * c2 * det_a * det_b
    return total


def _run_lih_casscf(bond_length):
    xyz = f"Li 0.0 0.0 0.0\nH  0.0 0.0 {bond_length}"
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4],
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()
    return system, mc


@pytest.mark.parametrize("backend", BACKENDS)
def test_casscf_wavefunction_overlap_two_independent_mcscf_runs(backend):
    """
    Two fully independent CASSCF (MCOptimizer) optimizations, at different
    bond lengths, so the resulting orbitals and CI vectors are unrelated by
    any known analytic transformation (unlike the "rotated active orbitals"
    test above). Each backend is cross-checked against an independent
    brute-force Loewdin determinant-overlap sum.
    """
    system_1, mc_1 = _run_lih_casscf(2.8)
    system_2, mc_2 = _run_lih_casscf(3.4)

    st1 = _ci_state(mc_1)
    st2 = _ci_state(mc_2)
    assert (st2.ndocc, st2.nactv) == (st1.ndocc, st1.nactv)

    S_pipeline = casscf_wavefunction_overlap(
        *st1.overlap_args(),
        system_1,
        *st2.overlap_args(),
        system_2,
        st1.ndocc,
        st1.nactv,
        backend=backend,
    )

    S_XY_full = mo_overlap(st1.C_docc_actv, system_1, st2.C_docc_actv, system_2)
    S_brute = _brute_force_determinant_overlap(
        S_XY_full,
        st1.ndocc,
        st1.ci_strings.make_determinants(),
        st1.C,
        st2.ci_strings.make_determinants(),
        st2.C,
    )

    # sanity: two different geometries' ground states must not be orthogonal
    # nor identical.
    assert 0.0 < abs(S_brute) < 1.0
    assert S_pipeline == approx_abs(S_brute, 1e-8)


def test_transform_ci_vector_direct_matches_sparse_ops():
    """
    Regression test proving transform_ci_vector_direct (string-addressed,
    efficient) implements the same math as transform_ci_vector_sparse_ops
    (generic sparse-operator, ground truth), for a random non-symmetric
    t_actv -- the exact case our generator produces (a matrix logarithm, not
    a Hermitian Hamiltonian).
    """
    _, ci = _lih_ci()
    st = _ci_state(ci)

    rng = np.random.default_rng(7)
    t_actv = rng.standard_normal((st.nactv, st.nactv)) * 0.1  # not symmetrized

    result_direct = transform_ci_vector_direct(st.ci_strings, st.C, t_actv)
    result_sparse_ops = transform_ci_vector_sparse_ops(st.ci_strings, st.C, t_actv)

    dets = st.ci_strings.make_determinants()
    result_sparse_ops_dense = _sparse_to_dense(result_sparse_ops, dets).real

    np.testing.assert_allclose(
        result_direct, result_sparse_ops_dense, rtol=1e-9, atol=1e-9
    )


def test_transform_ci_vector_direct_large_angle():
    """
    Exercise the scaling-and-squaring path with a deliberately large-norm
    (~3 radians) antisymmetric t_actv -- larger than the sparse-ops backend's
    *default* maxk/screen_thresh can converge (see pilot development history,
    where an angle of this order needed maxk raised to 32 with
    screen_thresh=1e-14 to reach machine precision). transform_ci_vector_direct
    must converge correctly with its own defaults, verified against
    transform_ci_vector_sparse_ops run with deliberately tight settings.
    """
    _, ci = _lih_ci()
    st = _ci_state(ci)

    rng = np.random.default_rng(11)
    A = rng.standard_normal((st.nactv, st.nactv))
    A = A - A.T  # antisymmetric generator: exp(-T) is unitary
    t_actv = 3.0 * A / np.linalg.norm(A, ord=2)  # rescale to spectral norm 3

    result_direct = transform_ci_vector_direct(st.ci_strings, st.C, t_actv)
    result_sparse_ops = transform_ci_vector_sparse_ops(
        st.ci_strings, st.C, t_actv, maxk=60, screen_thresh=1e-16
    )
    dets = st.ci_strings.make_determinants()
    result_sparse_ops_dense = _sparse_to_dense(result_sparse_ops, dets).real
    np.testing.assert_allclose(
        result_direct, result_sparse_ops_dense, rtol=1e-8, atol=1e-8
    )

    # sanity: exp(-T) for an antisymmetric T is unitary, so it must preserve
    # the CI vector's norm.
    assert np.linalg.norm(result_direct) == pytest.approx(1.0, abs=1e-10)


# -- Two-component (spinor) wavefunction overlap ----------------------------


def _rel_lih_ci():
    xyz = "Li 0.0 0.0 0.0\nH  0.0 0.0 3.0"
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-10)(system)
    ci = RelCI(nel=4, core_orbitals=2, active_orbitals=8)(scf)
    ci.run()
    return system, ci


@pytest.mark.parametrize("backend", BACKENDS)
def test_casscf_wavefunction_overlap_two_component_same_orbitals(backend):
    system, ci = _rel_lih_ci()
    st = _ci_state(ci)

    S = casscf_wavefunction_overlap(
        *st.overlap_args(),
        system,
        *st.overlap_args(),
        system,
        st.ndocc,
        st.nactv,
        backend=backend,
    )
    assert S == approx_abs(1.0, 1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_casscf_wavefunction_overlap_two_component_docc_rotation(backend):
    """
    Rotate ONLY the docc spinors by a random unitary ``U_docc``, leaving the
    active orbitals and the determinant-basis CI vector untouched. Two-component
    docc spinors are singly (not doubly) occupied, so the biorthogonalized
    overlap of the original wavefunction against this re-expressed one must
    equal exactly ``det(U_docc)`` -- not ``det(U_docc)**2``, the nonrelativistic
    power (see the docc_power note in casscf_wavefunction_overlap). This is a
    sharp, closed-form check on that power: get it wrong and this test's
    prediction misses by exactly a squaring.
    """
    system, ci = _rel_lih_ci()
    st = _ci_state(ci)

    rng = np.random.default_rng(3)
    A = rng.standard_normal((st.ndocc, st.ndocc)) + 1j * rng.standard_normal(
        (st.ndocc, st.ndocc)
    )
    A = A - A.conj().T  # anti-Hermitian generator: U_docc is unitary
    U_docc = expm(A)

    C_docc_actv_2 = st.C_docc_actv.copy()
    C_docc_actv_2[:, : st.ndocc] = st.C_docc_actv[:, : st.ndocc] @ U_docc

    S = casscf_wavefunction_overlap(
        *st.overlap_args(),
        system,
        st.ci_strings,
        st.C,
        C_docc_actv_2,
        system,
        st.ndocc,
        st.nactv,
        backend=backend,
    )
    assert S == approx_abs(np.linalg.det(U_docc), 1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_casscf_wavefunction_overlap_two_component_rotated_active_orbitals(backend):
    """
    Two-component analog of test_casscf_wavefunction_overlap_rotated_active_orbitals:
    rotate the active spinors by a random unitary R and independently build the
    CI vector of the SAME physical wavefunction in the rotated basis (via
    SparseExp/apply_op on the single-channel generator log(R), see
    _one_body_sparse_operator's two_component branch). The biorthogonalized
    overlap between the original and rotated representation must recover 1.
    """
    from forte2.lib.sparse_ops import SparseExp

    system, ci = _rel_lih_ci()
    st = _ci_state(ci)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((st.nactv, st.nactv)) + 1j * rng.standard_normal(
        (st.nactv, st.nactv)
    )
    A = A - A.conj().T  # anti-Hermitian generator: R is unitary
    R = expm(A)

    state1 = _make_sparse_state(st.ci_strings, st.C, 1e-14)
    T_op = _one_body_sparse_operator(logm(R), two_component=True)
    state2_ref = SparseExp(32, 1e-14).apply_op(T_op, state1, scaling_factor=-1.0)

    C2_det = _sparse_to_dense(state2_ref, st.ci_strings.make_determinants())

    C_docc_actv_2 = st.C_docc_actv.copy()
    C_docc_actv_2[:, st.ndocc :] = st.C_docc_actv[:, st.ndocc :] @ R

    S = casscf_wavefunction_overlap(
        *st.overlap_args(),
        system,
        st.ci_strings,
        C2_det,
        C_docc_actv_2,
        system,
        st.ndocc,
        st.nactv,
        backend=backend,
    )
    assert S == approx_abs(1.0, 1e-8)


def test_transform_ci_vector_direct_matches_sparse_ops_two_component():
    """
    Two-component analog of test_transform_ci_vector_direct_matches_sparse_ops:
    proves transform_ci_vector_direct's RelCISigmaBuilder-based path implements
    the same math as transform_ci_vector_sparse_ops, for a random non-Hermitian
    complex t_actv (the exact case our generator produces, a matrix logarithm,
    not a physical Hamiltonian) and a genuinely complex CI vector.
    """
    _, ci = _rel_lih_ci()
    st = _ci_state(ci)

    rng = np.random.default_rng(7)
    t_actv = (
        rng.standard_normal((st.nactv, st.nactv))
        + 1j * rng.standard_normal((st.nactv, st.nactv))
    ) * 0.1  # not Hermitian

    result_direct = transform_ci_vector_direct(
        st.ci_strings, st.C, t_actv, two_component=True
    )
    result_sparse_ops = transform_ci_vector_sparse_ops(
        st.ci_strings, st.C, t_actv, two_component=True
    )

    dets = st.ci_strings.make_determinants()
    result_sparse_ops_dense = _sparse_to_dense(result_sparse_ops, dets)

    np.testing.assert_allclose(
        result_direct, result_sparse_ops_dense, rtol=1e-9, atol=1e-9
    )


# -- Frozen core and input validation ---------------------------------------


def _run_fh_ci(shift):
    """FH with four docc orbitals, rigidly translated by `shift` bohr."""
    xyz = f"F 0.0 0.0 {shift}\nH 0.0 0.0 {1.73 + shift}"
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, e_tol=1e-11)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5],
    )(rhf)
    ci.run()
    return system, ci


@pytest.mark.parametrize("backend", BACKENDS)
def test_frozen_core_recovers_translated_overlap(backend):
    """
    A rigid translation leaves the wavefunction physically unchanged, so the
    overlap should be 1. It is not: core orbitals move with their nuclei, and
    their decaying mutual overlap multiplies straight into the result (Plasser
    Sec. 3.3). Discarding the tight 1s recovers most of the loss, and the
    result must not depend on which backend computes it.
    """
    system_0, ci_0 = _run_fh_ci(0.0)
    system_1, ci_1 = _run_fh_ci(0.1)
    st0, st1 = _ci_state(ci_0), _ci_state(ci_1)

    def overlap(n_frozen_docc):
        return abs(
            casscf_wavefunction_overlap(
                *st0.overlap_args(),
                system_0,
                *st1.overlap_args(),
                system_1,
                st0.ndocc,
                st0.nactv,
                backend=backend,
                n_frozen_docc=n_frozen_docc,
            )
        )

    all_cores = overlap(0)
    frozen_1s = overlap(1)

    # the artifact is large, and freezing the 1s removes most of it
    assert all_cores < 0.8
    assert frozen_1s > 0.9
    assert frozen_1s > all_cores


def test_frozen_core_warns_when_coupled_to_retained_orbitals():
    """
    Freezing is only valid for orbitals orthogonal to the retained space.
    Freezing FH's valence docc orbitals violates that and silently returns a
    badly wrong overlap, so it must warn; freezing only the 1s must not.
    """
    system_0, ci_0 = _run_fh_ci(0.0)
    system_1, ci_1 = _run_fh_ci(0.1)
    st0, st1 = _ci_state(ci_0), _ci_state(ci_1)
    ndocc, nactv = st0.ndocc, st0.nactv

    args = (*st0.overlap_args(), system_0, *st1.overlap_args(), system_1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        casscf_wavefunction_overlap(*args, ndocc, nactv, n_frozen_docc=1)

    with pytest.warns(UserWarning, match="coupled to the retained orbitals"):
        bad = casscf_wavefunction_overlap(*args, ndocc, nactv, n_frozen_docc=ndocc)
    assert abs(bad) < 0.2  # the warning is not hypothetical


def test_casscf_wavefunction_overlap_rejects_invalid_inputs():
    system, ci = _lih_ci()
    st = _ci_state(ci)
    ndocc, nactv = st.ndocc, st.nactv
    good = (*st.overlap_args(), system, *st.overlap_args(), system)

    rel_system, rel_ci = _rel_lih_ci()
    rel_st = _ci_state(rel_ci)

    with pytest.raises(ValueError, match="two_component"):
        casscf_wavefunction_overlap(
            *st.overlap_args(),
            system,
            *rel_st.overlap_args(),
            rel_system,
            ndocc,
            nactv,
        )

    with pytest.raises(ValueError, match="n_frozen_docc"):
        casscf_wavefunction_overlap(*good, ndocc, nactv, n_frozen_docc=ndocc + 1)

    with pytest.raises(ValueError, match="C1 has shape"):
        casscf_wavefunction_overlap(
            st.ci_strings,
            st.C[:-1],
            st.C_docc_actv,
            system,
            *st.overlap_args(),
            system,
            ndocc,
            nactv,
        )

    with pytest.raises(ValueError, match="columns"):
        casscf_wavefunction_overlap(
            st.ci_strings,
            st.C,
            st.C_docc_actv[:, :-1],
            system,
            *st.overlap_args(),
            system,
            ndocc,
            nactv,
        )
