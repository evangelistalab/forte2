import forte2, forte2.helpers
import numpy as np
import scipy, scipy.constants

X2C_LINDEP_TOL = 1e-10
LIGHT_SPEED = scipy.constants.physical_constants["inverse fine-structure constant"][0]
PAULI_MATRICES = np.array(
    [
        [[1, 0], [0, -1]],  # z
        [[0, 1], [1, 0]],  # x
        [[0, -1j], [1j, 0]],  # y
    ]
)


def _block_diag(A):
    return scipy.linalg.block_diag([A, A])


def _i_sigma_dot(A):
    nspnr = A.shape[-1] * 2
    quaterion = np.vstack([np.eye(2)[np.newaxis, :, :], 1j * PAULI_MATRICES])
    return np.einsum("xij,xmn->imjn", quaterion, A).reshape((nspnr,) * 2)


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
    c0 = LIGHT_SPEED

    S, T, V, W = _get_integrals(xbasis, system.atoms, x2c_type=x2c_type)

    # build the one-electron matrix Dirac equation
    D, M = build_dirac_eq(nbf, c0, S, T, V, W)

    # diagonalize the Dirac equation
    e_dirac, c_dirac = forte2.helpers.eigh_gen(
        D, M, tol=X2C_LINDEP_TOL, orth="symmetric"
    )

    # build the decoupling matrix X
    X = get_decoupling_matrix(c_dirac, nbf)

    # build the transformation matrix R
    R = get_transformation_matrix(c0, S, T, X, tol=X2C_LINDEP_TOL)

    # build the Foldy-Wouthuysen Hamiltonian
    h_fw = build_foldy_wouthuysen_hamiltonian(X, R, T, V, W, c0)

    # project back to the contracted basis
    h_fw = proj.conj().T @ h_fw @ proj

    return h_fw


def build_dirac_eq(nbf, c0, S, T, V, W):
    D = np.zeros((nbf * 2,) * 2)
    M = np.zeros((nbf * 2,) * 2)
    D[:nbf, :nbf] = V
    D[nbf:, nbf:] = (0.25 / c0**2) * W - T
    D[:nbf, nbf:] = T
    D[nbf:, :nbf] = T
    M[:nbf, :nbf] = S
    M[nbf:, nbf:] = (0.5 / c0**2) * T
    return D, M


def get_decoupling_matrix(c_dirac, nbf):
    clpos = c_dirac[:nbf, nbf:]
    cspos = c_dirac[nbf:, nbf:]
    # the two ways are equivalent
    # X = scipy.linalg.solve(clpos.T.conj(), cspos.T.conj()).T.conj()
    return cspos @ scipy.linalg.pinv(clpos)


def get_transformation_matrix(c0, S, T, X, tol=1e-9):
    S_tilde = S + (0.5 / c0**2) * X.conj().T @ T @ X
    Ssqrt = scipy.linalg.sqrtm(S)
    S12 = forte2.helpers.invsqrt_matrix(S, tol=tol)
    SSS = S12 @ S_tilde @ S12
    SSS12 = forte2.helpers.invsqrt_matrix(SSS, tol=tol)
    return S12 @ SSS12 @ Ssqrt


def build_foldy_wouthuysen_hamiltonian(X, R, T, V, W, c0):
    L = (
        T @ X
        + X.conj().T @ T
        - X.conj().T @ T @ X
        + V
        + (0.25 / c0**2) * X.conj().T @ W @ X
    )
    return R.conj().T @ L @ R
