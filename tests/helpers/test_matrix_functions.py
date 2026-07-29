import numpy as np
import scipy as sp

from forte2.helpers import invsqrt_matrix, eigh_gen, canonical_orth, random_unitary
from forte2.helpers.matrix_functions import compute_Am1y
from forte2.helpers.comparisons import approx


def test_eigh_gen_complex_hermitian():
    # Regression: eigh_gen transformed A with X.T instead of X.conj().T, giving
    # wrong eigenpairs for complex Hermitian generalized eigenproblems.
    rng = np.random.default_rng(0)
    n = 6
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M + M.conj().T
    N = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    B = N @ N.conj().T + n * np.eye(n)  # complex Hermitian positive-definite

    e_sp, _ = sp.linalg.eigh(A, B)
    e_ft, c_ft, _ = eigh_gen(A, B)

    assert np.allclose(np.sort(e_sp), np.sort(e_ft.real), atol=1e-8)
    # Each returned pair must satisfy A c = e B c.
    residual = np.linalg.norm(A @ c_ft - (B @ c_ft) * e_ft)
    assert residual < 1e-8


def test_compute_Am1y_complex_hermitian():
    # Regression: _compute_Am1y_eigh used evecs.T instead of evecs.conj().T,
    # giving a wrong A^{-1} y for complex Hermitian A.
    rng = np.random.default_rng(1)
    n = 5
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M @ M.conj().T + n * np.eye(n)  # complex Hermitian positive-definite
    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    x = compute_Am1y(A, y, ortho_rtol=1e-12)
    assert np.linalg.norm(A @ x - y) < 1e-8


def test_invsqrt_matrix():
    S = np.eye(10)
    S_od = np.random.rand(10, 10) * 0.05
    S += S_od + S_od.T
    Sm12, *_ = invsqrt_matrix(S, rtol=1e-10)
    assert np.allclose(Sm12 @ S @ Sm12, np.eye(10))

    Sm1_ref = np.linalg.inv(S)
    assert np.allclose(Sm12 @ Sm12, Sm1_ref)


def test_invsqrt_matrix_singular():
    S = np.ones((50, 50))
    Sm12, *_ = invsqrt_matrix(S, rtol=1e-10)
    pinv = np.linalg.pinv(S)
    # Sm12**2 should be the pseudo-inverse of S (S^+) in case of singular S
    assert np.allclose(pinv, Sm12 @ Sm12)
    # SS^+S = S (property of pseudo-inverse), but SS^+ is not necessarily identity
    assert np.allclose(S @ Sm12 @ Sm12 @ S, S)


def test_canonical_orth():
    # fix the seed for reproducibility
    generator = np.random.default_rng(42)
    H = generator.random((10, 10))
    H += H.T
    S = np.eye(10) + np.abs(generator.random((10, 10)) * 0.05)
    S = 0.5 * (S + S.T)

    X, Xm1, _ = canonical_orth(S, rtol=1e-10)
    assert np.allclose(X.T @ S @ X, np.eye(10))
    assert np.allclose(Xm1 @ X, np.eye(10))
    # X @ Xm1 is not necessarily identity

    e_sp, c_sp = sp.linalg.eigh(H, S)
    e_ft, c_ft, _ = eigh_gen(H, S)

    assert np.allclose(e_sp, e_ft)
    assert np.linalg.norm(c_sp @ c_sp.T - c_ft @ c_ft.T) < 1e-6


def test_random_unitary():
    rng = np.random.default_rng(42)
    for size in np.arange(10, 101, 10):
        U = random_unitary(size, cmplx=False, rng=rng, rotation=False)
        assert np.allclose(U.T @ U, np.eye(size))
        assert np.allclose(U @ U.T, np.eye(size))
        assert np.isclose(np.abs(np.linalg.det(U)), 1.0)
    for size in np.arange(10, 101, 10):
        U = random_unitary(size, cmplx=True, rng=rng, rotation=False)
        assert np.allclose(U.T.conj() @ U, np.eye(size))
        assert np.allclose(U @ U.T.conj(), np.eye(size))
        assert np.isclose(np.abs(np.linalg.det(U)), 1.0)
    for size in np.arange(10, 101, 10):
        U = random_unitary(size, cmplx=False, rng=rng, rotation=True)
        assert np.allclose(U.T @ U, np.eye(size))
        assert np.allclose(U @ U.T, np.eye(size))
        assert np.isclose(np.linalg.det(U), 1.0)
    for size in np.arange(10, 101, 10):
        U = random_unitary(size, cmplx=True, rng=rng, rotation=True)
        assert np.allclose(U.T.conj() @ U, np.eye(size))
        assert np.allclose(U @ U.T.conj(), np.eye(size))
        assert np.isclose(np.linalg.det(U), 1.0)


def test_symmetric_orth():
    # fix the seed for reproducibility
    generator = np.random.default_rng(42)
    H = generator.random((10, 10))
    H += H.T
    S = np.eye(10) + np.abs(generator.random((10, 10)) * 0.05)
    S = 0.5 * (S + S.T)
    e_sp, c_sp = sp.linalg.eigh(H, S)
    e_ft, c_ft, _ = eigh_gen(H, S, mode="symmetric")

    assert np.allclose(e_sp, e_ft)
    assert np.linalg.norm(c_sp @ c_sp.T - c_ft @ c_ft.T) < 1e-6


def test_canonical_orth_with_lindep():
    H = np.array([[1, 0.5], [0.5, 1]])
    S = np.array([[1, 1 - 1e-10], [1 - 1e-10, 1]])
    e, c, _ = eigh_gen(H, S)
    assert len(e) == 1
    assert e[0] == approx(0.75)
    assert c.flatten() == approx([0.5, 0.5])


def test_canonical_orth_with_lindep_2():
    # fix the seed for reproducibility
    rng = np.random.default_rng(42)
    size = 50

    s_eigh = rng.uniform(0.95, 1.05, size - 5)
    s_eigh = np.concatenate([s_eigh, 1e-12 * rng.uniform(0.5, 1.5, 5)])
    u_rand = random_unitary(size, cmplx=False, rng=rng, rotation=True)
    S = u_rand @ np.diag(s_eigh) @ u_rand.T

    X, Xm1, _ = canonical_orth(S, rtol=1e-10)
    assert np.allclose(X.T @ S @ X, np.eye(size - 5))
    assert np.allclose(Xm1 @ X, np.eye(size - 5))
