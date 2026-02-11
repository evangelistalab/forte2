import numpy as np
import scipy as sp

from forte2.helpers import invsqrt_matrix, eigh_gen, canonical_orth
from forte2.helpers.comparisons import approx


def test_invsqrt_matrix():
    S = np.eye(10)
    S_od = np.random.rand(10, 10) * 0.05
    S += S_od + S_od.T
    Sm12 = invsqrt_matrix(S, tol=1e-10)
    assert np.allclose(Sm12 @ S @ Sm12, np.eye(10))

    Sm1_ref = np.linalg.inv(S)
    assert np.allclose(Sm12 @ Sm12, Sm1_ref)


def test_invsqrt_matrix_singular():
    S = np.ones((50, 50))
    Sm12 = invsqrt_matrix(S, tol=1e-10)
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

    X, Xm1 = canonical_orth(S, tol=1e-10, return_inverse=True)
    assert np.allclose(X.T @ S @ X, np.eye(10))
    assert np.allclose(Xm1 @ X, np.eye(10))
    assert np.allclose(X @ Xm1, np.eye(10))

    e_sp, c_sp = sp.linalg.eigh(H, S)
    e_ft, c_ft = eigh_gen(H, S, remove_lindep=True, orth_method="canonical")

    assert np.allclose(e_sp, e_ft)
    assert np.linalg.norm(c_sp @ c_sp.T - c_ft @ c_ft.T) < 1e-6


def test_symmetric_orth():
    # fix the seed for reproducibility
    generator = np.random.default_rng(42)
    H = generator.random((10, 10))
    H += H.T
    S = np.eye(10) + np.abs(generator.random((10, 10)) * 0.05)
    S = 0.5 * (S + S.T)
    e_sp, c_sp = sp.linalg.eigh(H, S)
    e_ft, c_ft = eigh_gen(H, S, remove_lindep=True, orth_method="symmetric")

    assert np.allclose(e_sp, e_ft)
    assert np.linalg.norm(c_sp @ c_sp.T - c_ft @ c_ft.T) < 1e-6


def test_canonical_orth_with_lindep():
    H = np.array([[1, 0.5], [0.5, 1]])
    S = np.array([[1, 1 - 1e-10], [1 - 1e-10, 1]])
    e, c = eigh_gen(H, S, remove_lindep=True, orth_method="canonical")
    assert len(e) == 1
    assert e[0] == approx(0.75)
    assert c.flatten() == approx([0.5, 0.5])
