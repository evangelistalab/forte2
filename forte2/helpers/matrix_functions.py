import numpy as np
import scipy as sp

from . import logger

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


def canonical_orth(S, rtol=1e-7, print_info=False):
    r"""
    Compute the canonical orthogonalization given the metric matrix S.

    Parameters
    ----------
    S : NDArray
        Metric matrix (must be positive semi-definite).
    rtol : float, optional, default=1e-7
        Relative threshold t for which values below t * max_eigenvalue are treated as zero.
    print_info : bool, optional, default=False
        If True, print additional information about the eigenvalues and orthogonalization process,
        only if any eigenvalues are discarded.

    Returns
    -------
    X : NDArray
        The (possibly rectangular) canonical orthogonalization matrix X, such that ``X.T @ S @ X = I``.
    Xm1 : NDArray
        The inverse of the orthogonalization matrix, such that ``Xm1 @ X = I``.
    info : dict
        A dictionary containing additional information, including:
        - "max_eigenvalue": The largest eigenvalue of S.
        - "min_eigenvalue": The smallest eigenvalue of S.
        - "condition_number": The condition number of S (max_eigenvalue / min_eigenvalue).
        - "n_discarded": The number of eigenvalues discarded due to being below the threshold.
        - "n_kept": The number of eigenvalues kept.
        - "largest_discarded_eigenvalue": The largest eigenvalue that was discarded.
        - "smallest_kept_eigenvalue": The smallest eigenvalue that was kept.

    Notes
    -----
    The canonical orthogonalization is defined as follows:

    .. math::
        \mathbf{X}_{\eta} = \mathbf{U}_{\eta} \mathbf{s}^{-1/2}_{\eta},

    where :math:`\mathbf{U}_{\eta}` are the eigenvectors of :math:`\mathbf{S}` corresponding to eigenvalues larger than :math:`\eta`, :math:`\mathbf{s}^{-1/2}_{\eta}` is a diagonal matrix containing the inverse square roots of those eigenvalues, and :math:`\eta=\mathrm{rtol} \times \max(\mathbf{s})` is the threshold for discarding small eigenvalues.

    The resulting matrix :math:`\mathbf{X}_{\eta}` satisfies :math:`\mathbf{X}_{\eta}^{\dagger} \mathbf{S} \mathbf{X}_{\eta} = \mathbf{I}_{\eta}`. If any eigenvalues are discarded, the resulting :math:`\mathbf{X}_{\eta}` will be rectangular with fewer columns than rows, and the inverse :math:`\mathbf{X}_{\eta}^{-1}` will be a left-inverse satisfying :math:`\mathbf{X}_{\eta}^{-1} \mathbf{X}_{\eta} = \mathbf{I}_{\eta}` but not necessarily :math:`\mathbf{X}_{\eta} \mathbf{X}_{\eta}^{-1} = \mathbf{I}`.

    Raises
    ------
    ValueError
        If the matrix S is not positive semi-definite.
    """
    # Compute the inverse square root of S
    info = {}
    sevals, sevecs = np.linalg.eigh(S)
    max_seval = sevals[-1]
    info["max_eigenvalue"] = max_seval
    info["min_eigenvalue"] = sevals[0]
    info["condition_number"] = max_seval / sevals[0]
    if np.any(sevals < -MACHEPS):
        raise ValueError("Matrix must be positive semi-definite.")

    # indices equal and above discard_idx are kept
    ndiscard = np.searchsorted(sevals, rtol * max_seval)
    info["n_discarded"] = ndiscard
    info["n_kept"] = len(sevals) - ndiscard
    info["largest_discarded_eigenvalue"] = sevals[ndiscard - 1] if ndiscard > 0 else 0.0
    info["smallest_kept_eigenvalue"] = (
        sevals[ndiscard] if ndiscard < len(sevals) else 0.0
    )
    U = sevecs[:, ndiscard:]
    # X = U @ s^{-1/2}, so the s_i^{-1/2}'s scale the columns
    X = U / np.sqrt(sevals[ndiscard:])
    # X^{-1} = s^{1/2} @ U.+, so the s_i^{1/2}'s scale the rows
    Xm1 = np.sqrt(sevals[ndiscard:])[:, None] * U.T.conj()

    if print_info and info["n_discarded"] > 0:
        logger.log_info1("Canonical orthogonalization info:")
        logger.log_info1(f"  Max eigenvalue: {info['max_eigenvalue']:.4e}")
        logger.log_info1(f"  Min eigenvalue: {info['min_eigenvalue']:.4e}")
        logger.log_info1(f"  Condition number: {info['condition_number']:.4e}")
        logger.log_info1(f"  Number of discarded eigenvalues: {info['n_discarded']}")
        logger.log_info1(f"  Number of kept eigenvalues: {info['n_kept']}")
        logger.log_info1(
            f"  Largest discarded eigenvalue: {info['largest_discarded_eigenvalue']:.4e}"
        )
        logger.log_info1(
            f"  Smallest kept eigenvalue: {info['smallest_kept_eigenvalue']:.4e}"
        )

    return X, Xm1, info


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
            X, *_ = canonical_orth(B, orth_tol)
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


def random_unitary(size, cmplx=True, rng=None, rotation=True):
    """
    Generate a random orthogonal/unitary matrix of given size.

    Parameters
    ----------
    size : int
        The size of the matrix.
    cmplx : bool, optional, default=True
        If True, generate a complex unitary matrix; otherwise, generate a real orthogonal matrix.
    rng : np.random.Generator, optional
        A random number generator for reproducibility.
    rotation : bool, optional, default=True
        If True, return a proper rotation (determinant = 1) by adjusting the sign of the last column if necessary. If False, the determinant may be -1.

    Notes
    -----
    The QR of a random (not necessarily Hermitian) matrix with normally distributed entries can give an orthogonal/unitary matrix.
    However, due to the way QR works, the distribution of the resulting matrices is not uniform over O(n) or U(n).
    To ensure a uniform distribution (Haar measure), we need to adjust the signs/phases of the columns based on the diagonal of R.
    This method is commonly used to generate random unitary/orthogonal matrices that are uniformly distributed over the appropriate group (O(n) or U(n)).
    These matrices will have determinant ±1 for O(n) and determinant with magnitude 1 and arbitrary phase for U(n).
    For special groups (determinant = 1), we can further adjust the sign/phase of a single column (here chosen as the first column) to ensure the determinant is exactly 1, which gives us a uniform distribution over SO(n) or SU(n)).
    See more at https://case.edu/artsci/math/mwmeckes/elizabeth/Meckes_SAMSI_Lecture2.pdf

    Returns
    -------
    NDArray
        A random unitary (or orthogonal) matrix of shape (size, size).
        It is guaranteed to be uniformly distributed over the appropriate group ((S)O(n) or (S)U(n)).
    """
    if rng is None:
        rng = np.random.default_rng()

    if cmplx:
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, R = np.linalg.qr(A, mode="complete")

        d = np.diag(R)
        d = d / np.abs(d)  # unit phases (assumes no zeros)
        Q = Q * np.conj(d)  # scales columns by conjugate phases

        if rotation:  # SU(n)
            detQ = np.linalg.det(Q)
            Q[:, 0] *= np.conj(detQ)  # makes det exactly 1 (since |detQ|=1)
    else:
        A = rng.standard_normal((size, size))
        Q, R = np.linalg.qr(A, mode="complete")

        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        Q = Q * d  # scales columns

        if rotation:  # SO(n)
            sgn, _ = np.linalg.slogdet(Q)
            if sgn < 0:
                Q[:, 0] *= -1.0

    return Q


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
        The resulting matrix, with double the dimensions of the input arrays.
    """
    return np.block([[scalar + z * 1j, x * 1j + y], [x * 1j - y, scalar - z * 1j]])
