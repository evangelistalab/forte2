import numpy as np
import pytest
from scipy.linalg import expm, logm

from forte2 import System, RHF, CI, CISolver, MCOptimizer, State
from forte2.helpers.comparisons import approx_abs
from forte2.orbitals.orbital_overlap import mo_overlap
from forte2.orbitals.wavefunction_overlap import (
    biorthogonalize_casscf_orbitals,
    casscf_wavefunction_overlap,
    transform_ci_vector_direct,
    transform_ci_vector_sparse_ops,
    _one_body_sparse_operator,
)

BACKENDS = ["direct", "sparse_ops"]


def random_orthonormal(nbf, n, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((nbf, n)))
    return Q


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

    C_XA, C_YB = biorthogonalize_casscf_orbitals(S_XY, ndocc, nactv)

    S_AB = C_XA.conj().T @ S_XY @ C_YB
    assert np.max(np.abs(S_AB - np.eye(n))) == approx_abs(0.0, 1e-10)

    # the transformation must be block upper-triangular: new docc orbitals
    # are pure recombinations of old docc orbitals only.
    leak = C_XA[ndocc:, :ndocc]
    if leak.size:
        assert np.max(np.abs(leak)) == approx_abs(0.0, 1e-10)


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
    ci_solver = ci.sub_solvers[0]
    ndocc, nactv = ci.mo_space.ncore, ci.mo_space.nactv
    n = ndocc + nactv

    C_det = np.zeros(ci_solver.ndet)
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 0], C_det)
    C_docc_actv = ci.mos.C[0][:, ci.mo_space.orig_to_contig][:, :n]

    S = casscf_wavefunction_overlap(
        ci_solver.ci_strings,
        C_det,
        C_docc_actv,
        system,
        ci_solver.ci_strings,
        C_det,
        C_docc_actv,
        system,
        ndocc,
        nactv,
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
    ci_solver = ci.sub_solvers[0]
    ndocc, nactv = ci.mo_space.ncore, ci.mo_space.nactv
    n = ndocc + nactv

    C1_det = np.zeros(ci_solver.ndet)
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 0], C1_det)
    C_docc_actv_1 = ci.mos.C[0][:, ci.mo_space.orig_to_contig][:, :n]

    rng = np.random.default_rng(0)
    A = rng.standard_normal((nactv, nactv))
    R = expm(A - A.T)
    kappa = logm(R)

    state1 = ci_solver.ci_sigma_builder.make_sparse_state(C1_det, 1e-14)
    T_op = _one_body_sparse_operator(kappa)
    state2_ref = SparseExp(32, 1e-14).apply_op(T_op, state1, scaling_factor=-1.0)

    dets = ci_solver.ci_strings.make_determinants()
    C2_det = np.array([state2_ref[d].real if d in state2_ref else 0.0 for d in dets])

    C_docc_actv_2 = C_docc_actv_1.copy()
    C_docc_actv_2[:, ndocc:] = C_docc_actv_1[:, ndocc:] @ R

    S = casscf_wavefunction_overlap(
        ci_solver.ci_strings,
        C1_det,
        C_docc_actv_1,
        system,
        ci_solver.ci_strings,
        C2_det,
        C_docc_actv_2,
        system,
        ndocc,
        nactv,
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

    ss1 = mc_1.ci_solver.sub_solvers[0]
    ss2 = mc_2.ci_solver.sub_solvers[0]
    ndocc, nactv = mc_1.mo_space.ncore, mc_1.mo_space.nactv
    n = ndocc + nactv
    assert (mc_2.mo_space.ncore, mc_2.mo_space.nactv) == (ndocc, nactv)

    C1_det = np.zeros(ss1.ndet)
    ss1.spin_adapter.csf_C_to_det_C(ss1.evecs[:, 0], C1_det)
    C2_det = np.zeros(ss2.ndet)
    ss2.spin_adapter.csf_C_to_det_C(ss2.evecs[:, 0], C2_det)

    C_docc_actv_1 = mc_1.mos.C[0][:, mc_1.mo_space.orig_to_contig][:, :n]
    C_docc_actv_2 = mc_2.mos.C[0][:, mc_2.mo_space.orig_to_contig][:, :n]

    S_pipeline = casscf_wavefunction_overlap(
        ss1.ci_strings,
        C1_det,
        C_docc_actv_1,
        system_1,
        ss2.ci_strings,
        C2_det,
        C_docc_actv_2,
        system_2,
        ndocc,
        nactv,
        backend=backend,
    )

    S_XY_full = mo_overlap(C_docc_actv_1, system_1, C_docc_actv_2, system_2)
    dets1 = ss1.ci_strings.make_determinants()
    dets2 = ss2.ci_strings.make_determinants()
    S_brute = _brute_force_determinant_overlap(
        S_XY_full, ndocc, dets1, C1_det, dets2, C2_det
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
    ci_solver = ci.sub_solvers[0]
    nactv = ci.mo_space.nactv

    C_det = np.zeros(ci_solver.ndet)
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 0], C_det)

    rng = np.random.default_rng(7)
    t_actv = rng.standard_normal((nactv, nactv)) * 0.1  # not symmetrized

    result_direct = transform_ci_vector_direct(ci_solver.ci_strings, C_det, t_actv)
    result_sparse_ops = transform_ci_vector_sparse_ops(
        ci_solver.ci_strings, C_det, t_actv
    )

    dets = ci_solver.ci_strings.make_determinants()
    result_sparse_ops_dense = np.array(
        [result_sparse_ops[d].real if d in result_sparse_ops else 0.0 for d in dets]
    )

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
    ci_solver = ci.sub_solvers[0]
    nactv = ci.mo_space.nactv

    C_det = np.zeros(ci_solver.ndet)
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 0], C_det)

    rng = np.random.default_rng(11)
    A = rng.standard_normal((nactv, nactv))
    A = A - A.T  # antisymmetric generator: exp(-T) is unitary
    t_actv = 3.0 * A / np.linalg.norm(A, ord=2)  # rescale to spectral norm 3

    result_direct = transform_ci_vector_direct(ci_solver.ci_strings, C_det, t_actv)
    result_sparse_ops = transform_ci_vector_sparse_ops(
        ci_solver.ci_strings, C_det, t_actv, maxk=60, screen_thresh=1e-16
    )
    dets = ci_solver.ci_strings.make_determinants()
    result_sparse_ops_dense = np.array(
        [result_sparse_ops[d].real if d in result_sparse_ops else 0.0 for d in dets]
    )
    np.testing.assert_allclose(
        result_direct, result_sparse_ops_dense, rtol=1e-8, atol=1e-8
    )

    # sanity: exp(-T) for an antisymmetric T is unitary, so it must preserve
    # the CI vector's norm.
    assert np.linalg.norm(result_direct) == pytest.approx(1.0, abs=1e-10)
