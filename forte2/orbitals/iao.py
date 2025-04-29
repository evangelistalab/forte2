from forte2.helpers import invsqrt_matrix


def orthogonalize(M, S):
    """
    Orthogonalize the given matrix S using the Cholesky decomposition.

    Args:
        S (np.ndarray): The matrix to be orthogonalized.

    Returns:
        np.ndarray: The orthogonalized matrix.
    """

    X = C.T @ S @ C
    X_invsqrt = invsqrt_matrix(X)
    return C @ X_invsqrt


class IAO:
    """
    Class to represent the Inverse Atomic Orbital (IAO) basis set.
    """

    def run(self, system, C):
        """
        Run the IAO calculation for the given system.

        Args:
            system: The system for which the IAO is to be calculated.
        """

        basis = system.basis
        minao_basis = system.minao_basis
        if minao_basis is None:
            raise ValueError("No MINAO basis set found in the system.")

        S1 = forte2.ints.overlap(basis, basis)
        S12 = forte2.ints.overlap(basis, minao_basis)
        S2 = forte2.ints.overlap(minao_basis, minao_basis)

        S1_inv = np.linalg.pinv(S1)
        S2_inv = np.linalg.pinv(S2)
        P12 = S1_inv @ S12
        Ct = orthogonalize(S1_inv @ S12 @ S2_inv @ S12.T @ C)

        A = orthogonalize(
            C @ C.T @ S1 @ Ct @ Ct.T @ S1 @ P12
            + (np.eye(C.shape[0]) - C @ C.T @ S1)
            @ (np.eye(Ct.shape[0]) - Ct @ Ct.T @ S1)
            @ P12
        )
        return A
