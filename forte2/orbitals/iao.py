import numpy as np

from forte2 import ints
from forte2.system import System
from forte2.helpers import invsqrt_matrix, logger


class IAO:
    """
    Class to represent the intrinsic atomic orbital (IAO).

    Parameters
    ----------
    system : System
        The system for which the IAO is to be calculated.
    C_occ : NDArray
        The occupied molecular orbital coefficients.

    Attributes
    ----------
    C_iao : NDArray
        The orthonormalized IAO coefficients, shape (nbf, nminao).

    Notes
    -----
    JCTC 2013, 9, 4834-4843
    """

    def __init__(self, system: System, C_occ: np.ndarray):
        self.system = system
        self.C_occ = C_occ.copy()
        self.C_iao = self._make_iao(self.C_occ)
        self.nocc = C_occ.shape[1]

    def _make_iao(self, C):
        basis = self.system.basis
        nbf = self.system.nbf
        minao_basis = self.system.minao_basis
        if minao_basis is None:
            raise ValueError("No minao_basis found in the system.")

        # various overlap matrices, see appendix C of JCTC 2013, 9, 4834-4843
        self.S1 = ints.overlap(basis, basis)
        self.S12 = ints.overlap(basis, minao_basis)
        self.S2 = ints.overlap(minao_basis, minao_basis)

        S1_inv = np.linalg.pinv(self.S1)
        S2_inv = np.linalg.pinv(self.S2)

        # projector onto the large basis
        P12 = S1_inv @ self.S12
        # projector onto the minao basis
        P21 = S2_inv @ self.S12.T
        # downproject and upproject the occupied MOs to get a set of depolarized MOs
        # cf. eq 1
        C_depolarized = P12 @ P21 @ C
        # orthonormal set of depolarized MOs
        Ct = _orthogonalize(C_depolarized, self.S1)

        C_polarized_occ = C @ C.T @ self.S1 @ Ct @ Ct.T @ self.S1 @ P12
        C_polarized_vir = (
            (np.eye(nbf) - C @ C.T @ self.S1)
            @ (np.eye(nbf) - Ct @ Ct.T @ self.S1)
            @ P12
        )

        C_iao = _orthogonalize(C_polarized_occ + C_polarized_vir, self.S1)
        return C_iao

    def make_sf_1rdm(self, sf_1rdm_ao):
        r"""
        Generate the spin-free 1-particle density matrix in the IAO basis, given by

        .. math::
            \gamma_{\rho\sigma} = \langle\rho|\hat{\gamma}|\sigma\rangle,

        where :math:`\hat{\gamma}=2\sum_{i \in \text{occ}} |i\rangle\langle i|` is the
        closed-shell RHF 1e density matrix (see eq 3 in the JCTC paper).

        Parameters
        ----------
        sf_1rdm_ao : NDArray
            The spin-free 1-particle density matrix in the large AO basis.

        Returns
        -------
        NDArray
            The spin-free 1-particle density matrix in the IAO basis.
        """
        S = self.system.ints_overlap()
        # contracting two AO indices requires an intervening overlap matrix
        # due to the non-orthogonality of the AOs
        return self.C_iao.T @ (S @ sf_1rdm_ao @ S) @ self.C_iao


def _orthogonalize(C, S):
    """
    Given a subset of molecular orbitals C, shape (nbf, n), n<= nbf,
    orthonormalize among the n orbitals under the metric S,
    such that C_ortho.T @ S @ C_ortho = I.
    See appendix C of JCTC 2013, 9, 4834-4843
    """

    X = C.T @ S @ C
    X_invsqrt = invsqrt_matrix(X)
    return C @ X_invsqrt


class IBO(IAO):
    """
    Class to represent the intrinsic bond orbital basis.

    Parameters
    ----------
    system : System
        The system for which the IBO is to be calculated.
    C_occ : NDArray
        The occupied molecular orbital coefficients.
    spaces : list[list[int]], optional
        A list of lists of orbital indices, each list specifies a space,
        and the spaces are separately localized. E.g., core and active indices.
        The indices should cover all occupied orbitals exactly once.
    maxiter : int, optional, default=10
        The maximum number of iterations for the IBO optimization.
    gconv : float, optional, default=1e-8
        The RMS gradient convergence criterion for the IBO optimization.
    exponent : int, optional, default=4
        The exponent used in the IBO optimization, can be either 2 or 4.
        IBO maximizes sum of atomic electron occupation raised to the power of `exponent`.
        An exponent of 4 provides better description of delocalized systems.

    Notes
    -----
    There are typos in the original paper, specifically for the Aij and Bij elements.
    See the corrected paper at
    http://www.iboview.org/bin/iao_preprint.pdf
    also see the reference implementation at
    https://sites.psu.edu/knizia/software/
    """

    def __init__(
        self,
        system: System,
        C_occ: np.ndarray,
        spaces: list[list[int]] = None,
        maxiter=10,
        gconv=1e-8,
        exponent=4,
    ):
        super().__init__(system, C_occ)
        self.maxiter = maxiter
        self.gconv = gconv
        self.exponent = exponent
        if exponent not in (2, 4):
            raise ValueError("Exponent must be either 2 or 4.")
        if spaces is None:
            spaces = [list(range(self.nocc))]
        else:
            # assert spaces is a list of lists of integers
            if not all(
                isinstance(space, list) and all(isinstance(i, int) for i in space)
                for space in spaces
            ):
                raise ValueError("Spaces must be a list of lists of integers.")
            ind = set()
            for space in spaces:
                ind.update(space)
            assert len(ind) == self.nocc, "Spaces must cover all occupied orbitals."
            assert (
                sum(len(space) for space in spaces) == self.nocc
            ), "Each occupied orbital must appear in exactly one space (no duplicates or omissions)."

        self.C_ibo = []
        for space in spaces:
            self.C_ibo.append(self._make_ibo(space))
        self.C_ibo = np.hstack(self.C_ibo)

    def _make_ibo(self, space):
        # Occupied MO coefficients in the IAO basis
        # shape (nminao, len(space))
        C = self.C_occ[:, space].copy()
        nmo = len(space)
        C_occ_iao = self.C_iao.T @ self.S1 @ C
        center_first_and_last = self.system.minao_basis.center_first_and_last
        natoms = self.system.natoms

        ibo_iter = 0

        while ibo_iter < self.maxiter:
            grad = 0.0
            for i in range(nmo):
                for j in range(i):
                    # Bij is the gradient, Aij is the approximate Hessian
                    Aij = 0
                    Bij = 0
                    for iatom in range(natoms):
                        sl = slice(*center_first_and_last[iatom])
                        Qii = np.dot(C_occ_iao[sl, i], C_occ_iao[sl, i])
                        Qjj = np.dot(C_occ_iao[sl, j], C_occ_iao[sl, j])
                        Qij = np.dot(C_occ_iao[sl, i], C_occ_iao[sl, j])
                        if self.exponent == 2:
                            Aij += 4 * Qij**2 - (Qii - Qjj) ** 2
                            Bij += 4 * Qij * (Qii - Qjj)
                        elif self.exponent == 4:
                            Aij -= Qii**4 + Qjj**4
                            Aij += 6 * (Qii**2 + Qjj**2) * Qij**2
                            Aij += Qii**3 * Qjj + Qii * Qjj**3
                            Bij += 4 * Qij * (Qii**3 - Qjj**3)
                    grad += Bij**2
                    phi_ij = 0.25 * np.arctan2(Bij, -Aij)
                    i_new = (
                        np.cos(phi_ij) * C_occ_iao[:, i]
                        + np.sin(phi_ij) * C_occ_iao[:, j]
                    )
                    j_new = (
                        -np.sin(phi_ij) * C_occ_iao[:, i]
                        + np.cos(phi_ij) * C_occ_iao[:, j]
                    )
                    C_occ_iao[:, i] = i_new.copy()
                    C_occ_iao[:, j] = j_new.copy()
            if np.sqrt(grad) < self.gconv:
                logger.log_info1(f"\nIBO converged after {ibo_iter} iterations.")
                break
            ibo_iter += 1
        else:
            raise RuntimeError(
                f"IBO did not converge after {self.maxiter} iterations. Change `maxiter` or `gconv`."
            )

        # (nbf, nminao) @ (nminao, len(space)) = (nbf, len(space))
        return self.C_iao @ C_occ_iao
