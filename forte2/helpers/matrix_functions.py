import numpy as np
import scipy as sp

MACHEPS = 1e-14


def invsqrt_matrix(M, tol=1e-7):
    """
    Compute the inverse square root of a symmetric (Hermitian) matrix A.
    Small eigenvalues below 'tol' are treated as zero (pseudo-inverse style).

    Parameters
    ----------
    M : NDArray
        A symmetric matrix (must be positive semi-definite).
    tol : float, optional, default=1e-7
        Eigenvalue threshold below which values are treated as zero.

    Returns
    -------
    invsqrt_M : NDArray
        The inverse square root of A.

    Raises
    ------
    ValueError
        If the matrix M is not positive semi-definite.
    """
    # Symmetric eigenvalue decomposition
    evals, evecs = np.linalg.eigh(M)
    if np.any(evals < -MACHEPS):
        raise ValueError("Matrix must be positive semi-definite.")
    # Inverse sqrt eigenvalues with threshold
    invsqrt_evals = np.zeros_like(evals)
    for i, val in enumerate(evals):
        if val > tol:
            invsqrt_evals[i] = 1.0 / np.sqrt(val)
        else:
            invsqrt_evals[i] = 0.0  # treat small/singular values carefully

    # Rebuild the matrix
    invsqrt_M = evecs @ np.diag(invsqrt_evals) @ evecs.T.conj()
    return invsqrt_M


def canonical_orth(S, tol=1e-7, return_U=False):
    """
    Compute the canonical orthogonalization given the metric matrix S.

    Parameters
    ----------
    S : NDArray
        Metric matrix (must be positive semi-definite).
    tol : float, optional, default=1e-7
        Eigenvalue threshold below which values are treated as zero.
    return_U : bool, optional, default=False
        If True, also return the set of eigenvectors corresponding to non-zero eigenvalues.

    Returns
    -------
    X : NDArray
        The (possibly rectangular) canonical orthogonalization matrix X, such that ``X.T @ S @ X = I``.
    U : NDArray, optional
        The eigenvectors corresponding to non-zero eigenvalues, returned if `return_U` is True

    Raises
    ------
    ValueError
        If the matrix S is not positive semi-definite.
    """
    # Compute the inverse square root of S
    sevals, sevecs = np.linalg.eigh(S)
    if np.any(sevals < -MACHEPS):
        raise ValueError("Matrix must be positive semi-definite.")
    trunc_indices = np.where(sevals > tol)[0]
    U = sevecs[:, trunc_indices]
    X = U / np.sqrt(sevals[trunc_indices])
    if return_U:
        return X, U
    else:
        return X


def eigh_gen(A, B=None, remove_lindep=True, orth_tol=1e-7, orth_method="canonical"):
    """
    Solve the generalized eigenvalue problem ``A @ x = lambda * B @ x``.

    Parameters
    ----------
    A : NDArray
        The matrix A.
    B : NDArray
        The matrix B. If None, the identity matrix is used.
    remove_lindep : bool, optional, default=True
        If True, perform orthogonalization to remove linear dependencies, else use ``sp.linalg.eigh``.
    orth_tol : float, optional, default=1e-7
        Eigenvalue threshold below which values are treated as zero.
    orth_method : str, optional, default="canonical"
        Orthogonalization method. Options are "canonical" or "symmetric".
        "canonical" should be used when there are linear dependencies in the basis functions.

    Returns
    -------
    tuple(NDArray, NDArray)
        A tuple containing the eigenvalues and eigenvectors.
    """
    if B is None:
        B = np.eye(A.shape[0])

    if remove_lindep:
        if orth_method == "canonical":
            X = canonical_orth(B, orth_tol)
        elif orth_method == "symmetric":
            X = invsqrt_matrix(B, orth_tol)
        else:  # TODO: add partial cholesky: 10.1063/1.5139948
            raise ValueError("Invalid orthogonalization method.")

        A = X.T @ A @ X
        e, c = np.linalg.eigh(A)
        return e, X @ c
    else:
        return sp.linalg.eigh(A, B)


def givens_rotation(A, c, s, i, j, column=True):
    """
    Apply a Givens rotation to the matrix A.

    Parameters
    ----------
    A : NDArray
        The matrix to apply the rotation to.
    c : float
        The cosine of the rotation angle.
    s : float
        The sine of the rotation angle.
    i : int
        The index of the first row/column to rotate.
    j : int
        The index of the second row/column to rotate.
    column : bool, optional, default=True
        If True, apply the rotation to columns; if False, to rows.

    Returns
    -------
    NDArray
        The rotated matrix.
    """
    M = A.copy()
    if column:
        Ai = A[:, i]
        Aj = A[:, j]
        M[:, i] = c * Ai + np.conjugate(s) * Aj
        M[:, j] = -s * Ai + c * Aj
    else:
        Ai = A[i, :]
        Aj = A[j, :]
        M[i, :] = c * Ai - s * Aj
        M[j, :] = np.conjugate(s) * Ai + c * Aj
    return M


def cholesky_wrapper(M, tol):
    """
    Perform a Cholesky decomposition with complete pivoting, works with any symmetric positive semi-definite matrix.

    Parameters
    ----------
    M : NDArray
        The matrix to decompose.
    tol : float
        The tolerance for the decomposition.

    Returns
    -------
    B : NDArray
        The Cholesky factor such that ``B.T @ B = M``.
    """
    # dpstrf: Cholesky decomposition with complete pivoting
    # tol=-1 ~machine precision tolerance
    C, piv, rank, info = sp.linalg.lapack.dpstrf(M, tol=tol, lower=False)
    if info < 0:
        raise ValueError(
            f"dpstrf failed with info={info}, indicating the {-info}-th argument had an illegal value."
        )
    piv = piv - 1  # convert to 0-based indexing

    inv_piv = np.zeros_like(piv)
    inv_piv[piv] = np.arange(len(piv))

    B = np.triu(C)[:rank, inv_piv]
    return B

def block_diag_2x2(M, complex=True):
    """
    Return a block-diagonal matrix with two copies of `M` on the diagonal.
    Note this is **not** a function to block-diagonalize a matrix.

    Parameters
    ----------
    M : NDArray
        The matrix to convert, shape (n, n).
    complex : bool, optional, default=True
        If True, the output will be explicitly converted to complex type.

    Returns
    -------
    NDArray
        The block-diagonal matrix, shape (2n, 2n).
    """
    A = sp.linalg.block_diag(M, M)
    if complex:
        return A.astype(np.complex128)
    else:
        return A

def i_sigma_dot(scalar, x, y, z):
    """
    Construct the matrix i * (I2, sigma_x, sigma_y, sigma_z) dot (scalar, x, y, z).

    Parameters
    ----------
    scalar : ndarray
        The scalar component.
    x : ndarray
        The x component.
    y : ndarray
        The y component.
    z : ndarray
        The z component.

    Returns
    -------
    NDArray
        The 2x2 matrix representation.
    """
    return np.block([[scalar + z * 1j, x * 1j + y], [x * 1j - y, scalar - z * 1j]])
