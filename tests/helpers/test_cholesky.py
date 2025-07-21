import numpy as np
import pytest

from forte2 import System, ints
from forte2.helpers.matrix_functions import cholesky_wrapper


def test_cholesky_tei():
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=1e-10)
    eri = ints.coulomb_4c(system.basis)
    eri = eri.reshape((system.nbf**2,) * 2)
    B = cholesky_wrapper(eri, tol=1e-14)
    assert np.linalg.norm(B.T @ B - eri) < 1e-10


def test_cholesky_random_matrix():
    M = np.random.rand(10, 10)
    M = M @ M.T  # Make it symmetric positive semi-definite
    M[:, -1] = 0
    M[-1, :] = 0
    B = cholesky_wrapper(M, tol=1e-14)
    assert np.linalg.norm(B.T @ B - M) < 1e-10

    with pytest.raises(Exception):
        _ = np.linalg.cholesky(M)  # Should fail since M is not positive definite
