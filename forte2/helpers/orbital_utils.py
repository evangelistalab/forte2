import scipy as sp
import numpy as np


def check_orbital_orthonormality(C, S, tol=1e-8):
    """
    Check if a set of molecular orbitals is orthonormal.

    Parameters
    ----------
    C : ndarray
        Coefficient matrix of the molecular orbitals.
    S : ndarray
        Overlap matrix of the basis functions.
    tol : float
        Tolerance for checking orthonormality.

    Returns
    -------
    bool
        True if the orbitals are orthonormal, False otherwise.
    """
    ovlp = C.conj().T @ S @ C
    return np.allclose(ovlp, np.eye(ovlp.shape[0]), atol=tol, rtol=0)

def normalize_unitary_matrix(U, tol=1e-8):
    """
    Normalize a unitary matrix to ensure its columns are orthonormal.

    Parameters
    ----------
    U : ndarray
        Unitary matrix to be normalized.

    Returns
    -------
    ndarray
        Normalized unitary matrix.
    """
    is_unitary = np.allclose(U.conj().T @ U, np.eye(U.shape[1]), atol=tol, rtol=0)
    if not is_unitary:
        u, _, vh = sp.linalg.svd(U, full_matrices=False)
        return u @ vh
    else:
        return U