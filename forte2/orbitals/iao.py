import numpy as np
from dataclasses import dataclass

from forte2 import ints
from forte2.system import System
from forte2.helpers import invsqrt_matrix, logger


class IAO:
    """
    Class to represent the Inverse Atomic Orbital (IAO) basis set.

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
        self.C_iao = self._make_iao(C_occ.copy())

    def _make_iao(self, C):
        basis = self.system.basis
        nbf = self.system.nbf
        minao_basis = self.system.minao_basis
        if minao_basis is None:
            raise ValueError("No minao_basis found in the system.")

        # various overlap matrices, see appendix C of JCTC 2013, 9, 4834-4843
        S1 = ints.overlap(basis, basis)
        S12 = ints.overlap(basis, minao_basis)
        S2 = ints.overlap(minao_basis, minao_basis)

        S1_inv = np.linalg.pinv(S1)
        S2_inv = np.linalg.pinv(S2)

        # projector onto the large basis
        P12 = S1_inv @ S12
        # projector onto the minao basis
        P21 = S2_inv @ S12.T
        # downproject and upproject the occupied MOs to get a set of depolarized MOs
        # cf. eq 1
        C_depolarized = P12 @ P21 @ C
        # orthonormal set of depolarized MOs
        Ct = _orthogonalize(C_depolarized, S1)

        C_polarized_occ = C @ C.T @ S1 @ Ct @ Ct.T @ S1 @ P12
        C_polarized_vir = (
            (np.eye(nbf) - C @ C.T @ S1) @ (np.eye(nbf) - Ct @ Ct.T @ S1) @ P12
        )

        C_iao = _orthogonalize(C_polarized_occ + C_polarized_vir, S1)
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
    """See appendix C of JCTC 2013, 9, 4834-4843"""

    X = C.T @ S @ C
    X_invsqrt = invsqrt_matrix(X)
    return C @ X_invsqrt
