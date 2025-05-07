import forte2, forte2.helpers
import numpy as np
import scipy, scipy.constants
import functools

X2C_LINDEP_TOL = 1e-9
LIGHT_SPEED = scipy.constants.physical_constants["inverse fine-structure constant"][0]


def get_hcore_x2c(system, x2c_type="sf"):
    assert x2c_type in [
        "sf",
        "so",
    ], f"Invalid x2c_type: {x2c_type}. Must be 'sf' or 'so'."

    print("Number of contracted basis functions: ", system.nao())
    xbasis = system.decontract()
    nao = len(xbasis)
    print(f"Number of decontracted basis functions: {nao}")
    nbf = nao if x2c_type == "sf" else nao * 2
    # expensive way to get this for now but works for all types of contraction schemes
    proj = _get_projection_matrix(xbasis, system.basis, x2c_type=x2c_type)

    S, T, V, W = _get_integrals(xbasis, system.atoms, x2c_type=x2c_type)

    # build the one-electron matrix Dirac equation
    D, M = _build_dirac_eq(S, T, V, W, nbf, x2c_type)

    # diagonalize the Dirac equation
    e_dirac, c_dirac = forte2.helpers.eigh_gen(
        D, M, tol=X2C_LINDEP_TOL, orth="symmetric"
    )

    # build the decoupling matrix X
    X = _get_decoupling_matrix(c_dirac, nbf)

    # build the transformation matrix R
    R = _get_transformation_matrix(S, T, X, tol=X2C_LINDEP_TOL)

    # build the Foldy-Wouthuysen Hamiltonian
    h_fw = _build_foldy_wouthuysen_hamiltonian(X, R, T, V, W)

    # project back to the contracted basis
    h_fw = proj.conj().T @ h_fw @ proj

    return h_fw


def _block_diag(A):
    return scipy.linalg.block_diag(A, A)


def _i_sigma_dot(A):
    scalar, z, x, y = A
    return np.block([[scalar + z * 1j, x * 1j + y], [x * 1j - y, scalar - z * 1j]])


def _get_projection_matrix(xbasis, basis, x2c_type):
    proj = scipy.linalg.solve(
        forte2.ints.overlap(xbasis),
        forte2.ints.overlap(xbasis, basis),
        assume_a="pos",
    )
    return proj if x2c_type == "sf" else _block_diag(proj)


def _get_integrals(xbasis, atoms, x2c_type):
    S = forte2.ints.overlap(xbasis)
    T = forte2.ints.kinetic(xbasis)
    V = forte2.ints.nuclear(xbasis, atoms)
    W = forte2.ints.opVop(xbasis, atoms)
    if x2c_type == "sf":
        return S, T, V, W[0]
    elif x2c_type == "so":
        return _block_diag(S), _block_diag(T), _block_diag(V), _i_sigma_dot(W)


def _build_dirac_eq(S, T, V, W, nbf, x2c_type):
    dtype = np.float64 if x2c_type == "sf" else np.complex128
    D = np.zeros((nbf * 2,) * 2, dtype=dtype)
    M = np.zeros((nbf * 2,) * 2, dtype=dtype)
    D[:nbf, :nbf] = V
    D[nbf:, nbf:] = (0.25 / LIGHT_SPEED**2) * W - T
    D[:nbf, nbf:] = T
    D[nbf:, :nbf] = T
    M[:nbf, :nbf] = S
    M[nbf:, nbf:] = (0.5 / LIGHT_SPEED**2) * T
    return D, M


def _get_decoupling_matrix(c_dirac, nbf):
    clpos = c_dirac[:nbf, nbf:]
    cspos = c_dirac[nbf:, nbf:]
    return cspos @ scipy.linalg.pinv(clpos)


def _get_transformation_matrix(S, T, X, tol=1e-9):
    """
    This implementation follows eqs 26-34 of J. Chem. Phys. 131, 031104 (2009),
    which avoids doing matrix inversions and leads to a more numerically stable transformation.
    """
    S_tilde = S + (0.5 / LIGHT_SPEED**2) * X.conj().T @ T @ X
    lam, z = forte2.helpers.eigh_gen(S_tilde, S, tol=tol, orth="symmetric")
    idx = lam > 1e-14
    R = (z[:, idx] / np.sqrt(lam[idx])) @ z[:, idx].T.conj() @ S
    return R
    # This was the old way (Cheng and Gauss), worked fine for sfx2c1e, but seems unusable for sox2c1e
    # S_tilde = S + (0.5 / c0**2) * X.conj().T @ T @ X
    # Ssqrt = scipy.linalg.sqrtm(S)
    # S12 = forte2.helpers.invsqrt_matrix(S, tol=tol)
    # SSS = S12 @ S_tilde @ S12
    # SSS12 = forte2.helpers.invsqrt_matrix(SSS, tol=tol)
    # return S12 @ SSS12 @ Ssqrt


def _build_foldy_wouthuysen_hamiltonian(X, R, T, V, W):
    L = (
        T @ X
        + X.conj().T @ T
        - X.conj().T @ T @ X
        + V
        + (0.25 / LIGHT_SPEED**2) * X.conj().T @ W @ X
    )
    return R.conj().T @ L @ R
