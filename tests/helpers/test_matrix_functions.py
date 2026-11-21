import itertools

import numpy as np
import pytest
import scipy as sp

from forte2.helpers import (
    invsqrt_matrix,
    eigh_gen,
    canonical_orth,
    random_unitary,
    real_orthogonal_logm,
    split_unitary,
)
from forte2.helpers.comparisons import approx


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


def test_real_orthogonal_logm():
    rng = np.random.default_rng(42)
    rotations = [random_unitary(n, cmplx=False, rng=rng) for n in (1, 2, 7, 20)]
    # eigenvalues at -1 in a random basis, and an angle just short of pi, where
    # scipy's logm returns a complex result
    Z = random_unitary(6, cmplx=False, rng=rng)
    rotations.append(Z @ np.diag([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]) @ Z.T)
    c, s = np.cos(np.pi - 1e-9), np.sin(np.pi - 1e-9)
    W = random_unitary(3, cmplx=False, rng=rng)
    rotations.append(W @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ W.T)
    for Q in rotations:
        K = real_orthogonal_logm(Q)
        assert np.isrealobj(K)
        assert np.allclose(K, -K.T)
        assert np.allclose(sp.linalg.expm(K), Q)
    with pytest.raises(ValueError):
        real_orthogonal_logm(np.diag([-1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        real_orthogonal_logm(np.ones((4, 4)))
    with pytest.raises(ValueError):
        real_orthogonal_logm(random_unitary(6, cmplx=True, rng=rng))
    with pytest.raises(ValueError):
        real_orthogonal_logm(random_unitary(6, cmplx=False, rng=rng)[:, :4])


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


def test_split_unitary():
    rng = np.random.default_rng(67)

    def check(u, m, r):
        n = len(u)
        assert np.allclose(m @ r, u)
        # m has one element of unit modulus in each row and column
        nonzero = np.abs(m) > 1e-12
        assert np.all(nonzero.sum(axis=0) == 1)
        assert np.all(nonzero.sum(axis=1) == 1)
        assert np.allclose(np.abs(m[nonzero]), 1.0)
        assert np.allclose(r.conj().T @ r, np.eye(n))
        # r's diagonal holds the matched weights, real and non-negative, except
        # that in the real case the weakest can be flipped to make r proper
        diag = np.diag(r)
        assert np.allclose(diag.imag, 0.0)
        negative = np.flatnonzero(diag.real < 0)
        if np.isrealobj(u):
            assert np.isrealobj(m) and np.isrealobj(r)
            assert np.linalg.det(r) == approx(1.0)
            assert len(negative) <= 1
            if len(negative) == 1:
                assert negative[0] == np.argmin(np.abs(diag))
        else:
            assert len(negative) == 0

    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    m, r = split_unitary(A)
    assert np.allclose(m, A)
    assert np.allclose(r, np.eye(2))

    for cmplx in (False, True):
        # a phased permutation times a rotation close to the identity is split
        # back into its factors, up to the phases of the rotation's diagonal
        for n in (3, 10, 20):
            if cmplx:
                phases = np.exp(1j * rng.uniform(-np.pi, np.pi, n))
                K = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            else:
                phases = rng.choice([-1.0, 1.0], n)
                K = rng.standard_normal((n, n))
            P = np.eye(n)[:, rng.permutation(n)] * phases
            R = sp.linalg.expm(0.1 * (K - K.conj().T) / np.linalg.norm(K))
            u = P @ R
            m, r = split_unitary(u)
            check(u, m, r)
            d = np.diag(R) / np.abs(np.diag(R))
            assert np.allclose(m, P * d)
            assert np.allclose(r, d.conj()[:, None] * R)

        # for random unitaries, no other matching of columns to rows has a
        # larger total weight
        for n in (3, 4, 5):
            u = random_unitary(n, cmplx=cmplx, rng=rng, rotation=False)
            m, r = split_unitary(u)
            check(u, m, r)
            best = max(
                sum(abs(u[p[j], j]) for j in range(n))
                for p in itertools.permutations(range(n))
            )
            assert np.sum(np.abs(np.diag(r))) == approx(best)

        for n in (10, 20, 100):
            u = random_unitary(n, cmplx=cmplx, rng=rng, rotation=False)
            check(u, *split_unitary(u))

    with pytest.raises(ValueError):
        split_unitary(np.ones((3, 2)))
    with pytest.raises(ValueError):
        split_unitary(2.0 * np.eye(3))
