import forte2, forte2.helpers
import numpy as np
import scipy, scipy.constants


LIGHT_SPEED = scipy.constants.physical_constants["inverse fine-structure constant"][0]


def get_hcore_sfx2c1e(system):
    print("Number of contracted basis functions: ", system.nao())
    xbasis = system.decontract()
    nao = len(xbasis)
    print(f"Number of decontracted basis functions: {nao}")
    # expensive way to get this for now but works for all types of contraction schemes
    contr_coeff = scipy.linalg.solve(
        forte2.ints.overlap(xbasis),
        forte2.ints.overlap(xbasis, system.basis),
        assume_a="pos",
    )
    c0 = LIGHT_SPEED

    # get the integrals
    S = forte2.ints.overlap(xbasis)
    T = forte2.ints.kinetic(xbasis)
    V = forte2.ints.nuclear(xbasis, system.atoms)
    W = forte2.ints.opVop(xbasis, system.atoms)[0]

    # build the one-electron matrix Dirac equation
    D = np.zeros((nao * 2,) * 2)
    M = np.zeros((nao * 2,) * 2)
    D[:nao, :nao] = V
    D[nao:, nao:] = (0.25 / c0**2) * W - T
    D[:nao, nao:] = T
    D[nao:, :nao] = T
    M[:nao, :nao] = S
    M[nao:, nao:] = (0.5 / c0**2) * T

    # diagonalize the Dirac equation
    e_dirac, c_dirac = forte2.helpers.eigh_gen(D, M, tol=1e-9, orth="symmetric")

    # build the decoupling matrix X
    clpos = c_dirac[:nao, nao:]
    cspos = c_dirac[nao:, nao:]
    # the two ways are equivalent
    # X = scipy.linalg.solve(clpos.T.conj(), cspos.T.conj()).T.conj()
    X = cspos @ scipy.linalg.pinv(clpos)

    # build the transformation matrix R
    S_tilde = S + (0.5 / c0**2) * X.conj().T @ T @ X
    Ssqrt = scipy.linalg.sqrtm(S)
    S12 = forte2.helpers.invsqrt_matrix(S, tol=1e-9)
    SSS = S12 @ S_tilde @ S12
    SSS12 = forte2.helpers.invsqrt_matrix(SSS, tol=1e-9)
    R = S12 @ SSS12 @ Ssqrt

    # build the Foldy-Wouthuysen Hamiltonian
    L = (
        T @ X
        + X.conj().T @ T
        - X.conj().T @ T @ X
        + V
        + (0.25 / c0**2) * X.conj().T @ W @ X
    )
    h_fw = R.conj().T @ L @ R

    # project back to the contracted basis
    h_fw = contr_coeff.conj().T @ h_fw @ contr_coeff

    return h_fw
