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
    evals, evecs = np.linalg.eigh(M)
    if np.any(evals < 0):
        raise ValueError("Matrix must be positive semi-definite.")
    max_eval = np.max(np.abs(evals))
    # Inverse sqrt eigenvalues with threshold
    invsqrt_evals = np.zeros_like(evals)
    for i, val in enumerate(evals):
        if val > tol * max_eval:
            invsqrt_evals[i] = 1.0 / np.sqrt(val)
        else:
            invsqrt_evals[i] = 0.0  # treat small/singular values carefully

    # Rebuild the matrix
    invsqrt_M = evecs @ np.diag(invsqrt_evals) @ evecs.T
    return invsqrt_M
