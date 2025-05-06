import numpy as np
import scipy.linalg

MACHEPS = np.finfo(float).eps


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
    if np.any(evals < -MACHEPS):
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


def canonical_orth(S, tol=1e-7):
    """
    Compute the canonical orthogonalization of a symmetric matrix S.

    Args:
        S (np.ndarray): S symmetric matrix.
        tol (float): Eigenvalue threshold below which values are treated as zero.

    Returns:
        np.ndarray: The (possibly rectangular) canonical orthogonalization matrix X, such that X.T @ S @ X = I.
    """
    # Compute the inverse square root of S
    sevals, sevecs = np.linalg.eigh(S)
    if np.any(sevals < 0):
        raise ValueError("Matrix must be positive semi-definite.")
    max_eval = np.max(np.abs(sevals))
    trunc_indices = np.where(sevals > tol * max_eval)[0]
    X = sevecs[:, trunc_indices] / np.sqrt(sevals[trunc_indices])
    return X


def eigh_gen(A, B, tol=1e-7, orth="canonical"):
    """
    Solve the generalized eigenvalue problem A @ x = lambda * B @ x.

    Args:
        A (np.ndarray): The matrix A.
        B (np.ndarray): The matrix B.
        tol (float): Eigenvalue threshold below which values are treated as zero.
        orth (str): Orthogonalization method. Options are "canonical" or "symmetric".

    Returns:
        tuple: A tuple containing the eigenvalues and eigenvectors.
    """
    try:
        return scipy.linalg.eigh(A, B)
    except scipy.linalg.LinAlgError:
        print(
            f"Linear dependency detected in the generalized eigenvalue problem! Using orthogonalization method {orth}."
        )
        if orth == "canonical":
            X = canonical_orth(B, tol)
        elif orth == "symmetric":
            X = invsqrt_matrix(B, tol)
        else:
            raise ValueError("Invalid orthogonalization method.")

        A = X.T @ A @ X
        e, c = np.linalg.eigh(A)
        return e, X @ c
