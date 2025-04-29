import numpy as np


def invsqrt_matrix(M, tol=1e-7):
    """
    Compute the inverse square root of a symmetric (Hermitian) matrix A.
    Small eigenvalues below 'tol' are treated as zero (pseudo-inverse style).

    Args:
        M (np.ndarray): M symmetric matrix.
        tol (float): Eigenvalue threshold below which values are treated as zero.

    Returns:
        np.ndarray: The inverse square root of A.
    """
    # Symmetric eigenvalue decomposition
    evals, evecs = np.linalg.eigh(A)

    # Inverse sqrt eigenvalues with threshold
    invsqrt_evals = np.zeros_like(eigvals)
    for i, val in enumerate(eigvals):
        if val > tol:
            invsqrt_eigvals[i] = 1.0 / np.sqrt(val)
        else:
            invsqrt_eigvals[i] = 0.0  # treat small/singular values carefully

    # Rebuild the matrix
    invsqrt_M = eigvecs @ np.diag(invsqrt_eigvals) @ eigvecs.T
    return invsqrt_M
