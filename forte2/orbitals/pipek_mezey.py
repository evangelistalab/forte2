import numpy as np


def pipek_mezey_rotation(Q, exponent, maxiter, g_tol):
    """
    Find the rotation that localizes a set of orbitals by Pipek-Mezey Jacobi sweeps.

    The rotation U maximizes sum_A sum_i (Q^A_ii)^p, where Q^A = U^T Q^A U is the charge
    matrix of atom A in the rotated orbitals. For p = 2, each orbital pair is rotated by
    the exact optimal angle of Pipek and Mezey, JCP 90, 4916 (1989). For p = 4, it's
    rotated by the angle of Knizia, JCTC 9, 4834 (2013), Appendix D. The published
    Appendix has typos in the A_ij and B_ij elements; see the corrected preprint at
    http://www.iboview.org/bin/iao_preprint.pdf.

    Parameters
    ----------
    Q : NDArray
        The charge matrix of each atom, shape (natom, norb, norb). Each Q^A is symmetric.
    exponent : int
        The localization power p, 2 or 4.
    maxiter : int
        The maximum number of sweeps over all orbital pairs.
    g_tol : float
        The convergence threshold on the largest element of the orbital-rotation gradient.

    Returns
    -------
    U : NDArray
        The orthogonal rotation, shape (norb, norb).
    Q : NDArray
        The charge matrices of the rotated orbitals.
    nsweep : int
        The number of sweeps performed.

    Raises
    ------
    RuntimeError
        If the gradient is still above g_tol after maxiter sweeps.
    """
    Q = Q.copy()
    norb = Q.shape[1]
    U = np.eye(norb)
    nsweep = 0
    while _max_gradient(Q, exponent) >= g_tol:
        if nsweep == maxiter:
            raise RuntimeError(
                f"Orbital localization did not converge in {maxiter} sweeps. "
                "Increase maxiter or g_tol."
            )
        for i in range(norb):
            for j in range(i):
                rot = _jacobi_rotation(Q[:, i, i], Q[:, j, j], Q[:, i, j], exponent)
                Q[:, :, [i, j]] = Q[:, :, [i, j]] @ rot
                Q[:, [i, j], :] = rot.T @ Q[:, [i, j], :]
                U[:, [i, j]] = U[:, [i, j]] @ rot
        nsweep += 1
    return U, Q, nsweep


def _jacobi_rotation(qii, qjj, qij, exponent):
    if exponent == 2:
        A = np.sum(4 * qij**2 - (qii - qjj) ** 2)
        B = np.sum(4 * qij * (qii - qjj))
    else:
        A = np.sum(
            -(qii**4)
            - qjj**4
            + 6 * (qii**2 + qjj**2) * qij**2
            + qii**3 * qjj
            + qii * qjj**3
        )
        B = np.sum(4 * qij * (qii**3 - qjj**3))
    phi = 0.25 * np.arctan2(B, -A)
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def _max_gradient(Q, exponent):
    # dL/dphi_ij at phi = 0 is 2p sum_A Q^A_ij ((Q^A_ii)^(p-1) - (Q^A_jj)^(p-1))
    d = np.einsum("aii->ai", Q) ** (exponent - 1)
    G = 2 * exponent * np.einsum("aij,aij->ij", Q, d[:, :, None] - d[:, None, :])
    return np.abs(G).max()
