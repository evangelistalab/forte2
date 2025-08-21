import numpy as np
import scipy as sp
import itertools

from forte2 import ints
from forte2.system import System, ModelSystem
from forte2.helpers import logger
from forte2.helpers.matrix_functions import cholesky_wrapper


class FockBuilder:
    """
    Class to build the Fock matrix using the Cholesky decomposition of the auxiliary basis integrals.
    This class computes the atomic Coulomb (J) and exchange (K) matrices
    using the auxiliary basis functions.

    Parameters
    ----------
    system : System or ModelSystem
        The system for which to build the Fock matrix.
        If a ModelSystem is provided, it will decompose the 4D ERI tensor using Cholesky decomposition with complete pivoting.
    use_aux_corr : bool, optional, default=False
        If True, uses ``system.auxiliary_basis_set_corr`` instead of ``system.auxiliary_basis``.
    """

    def __init__(self, system: System, use_aux_corr=False):
        if isinstance(system, ModelSystem):
            # special handling for ModelSystem
            nbf = system.nbf
            eri = system.eri.reshape((nbf**2,) * 2)
            self.B = cholesky_wrapper(eri, tol=-1)
            self.B = self.B.reshape((self.B.shape[0], nbf, nbf))
            system.naux = self.B.shape[0]
            self.nbf = nbf
            self.naux = system.naux
            return

        if not system.cholesky_tei:
            basis = system.basis
            if use_aux_corr:
                assert hasattr(
                    system, "auxiliary_basis_set_corr"
                ), "The system does not have an auxiliary_basis_set_corr defined."
                aux_basis = system.auxiliary_basis_set_corr
            else:
                assert hasattr(
                    system, "auxiliary_basis"
                ), "The system does not have an auxiliary_basis defined."
                aux_basis = system.auxiliary_basis
            self.B = self._build_B_density_fitting(basis, aux_basis)
        else:
            self.B, system.naux = self._build_B_cholesky(
                system.basis, system.cholesky_tol
            )
        self.naux = system.naux
        self.nbf = system.nbf

    @staticmethod
    def _build_B_cholesky(basis, cholesky_tol):
        # Compute the memory requirements
        nbf = basis.size
        memory_gb = 8 * (nbf**4) / (1024**3)
        logger.log_info1("Building B tensor using Cholesky decomposition")
        logger.log_info1(
            f"Temporary memory requirement for 4-index integrals: {memory_gb:.2f} GB"
        )
        eri_full = ints.coulomb_4c(basis)
        eri = eri_full.reshape((nbf**2,) * 2)

        B = cholesky_wrapper(eri, tol=cholesky_tol)
        B = B.reshape((B.shape[0], nbf, nbf))

        naux = B.shape[0]

        memory_gb = 8 * (naux * nbf**2) / (1024**3)
        logger.log_info1(f"Memory requirements: {memory_gb:.2f} GB")
        logger.log_info1(f"Number of system basis functions: {nbf}")
        logger.log_info1(f"Number of auxiliary basis functions: {naux}")

        return B, naux

    @staticmethod
    def _build_B_density_fitting(basis, auxiliary_basis):
        # Compute the memory requirements
        nb = basis.size
        naux = auxiliary_basis.size
        memory_gb = 8 * (naux**2 + naux * nb**2) / (1024**3)
        logger.log_info1(f"Memory requirements: {memory_gb:.2f} GB")
        logger.log_info1(f"Number of system basis functions: {nb}")
        logger.log_info1(f"Number of auxiliary basis functions: {naux}")

        # Compute the integrals (P|Q) with P, Q in the auxiliary basis
        M = ints.coulomb_2c(auxiliary_basis, auxiliary_basis)

        # Decompose M = L L.T
        L = sp.linalg.cholesky(M)

        # Solve L.T X = I, or X = L.T^{-1} = M^{-1/2}
        I = np.eye(M.shape[0])
        M_inv_sqrt = sp.linalg.solve_triangular(L.T, I, lower=True)

        # Compute the integrals (P|mn) with P in the auxiliary basis and m, n in the system basis
        Pmn = ints.coulomb_3c(auxiliary_basis, basis, basis)

        # Compute B[P|mn] = M^{-1/2}[P|Q] (Q|mn)
        B = np.einsum("PQ,Qmn->Pmn", M_inv_sqrt, Pmn, optimize=True)
        del Pmn

        return B

    def build_J(self, D):
        J = [np.einsum("Pmn,Prs,sr->mn", self.B, self.B, Di, optimize=True) for Di in D]
        return J

    def build_K(self, C, cross=False):
        Y = [np.einsum("Pmr,mi->Pri", self.B, Ci.conj(), optimize=True) for Ci in C]
        if cross:
            K = []
            for Yi in Y:
                for Yj in Y:
                    K.append(np.einsum("Pmi,Pni->mn", Yi.conj(), Yj, optimize=True))
        else:
            K = [np.einsum("Pmi,Pni->mn", Yi.conj(), Yi, optimize=True) for Yi in Y]
        return K

    def build_K_density(self, D):
        K = [np.einsum("Pms,Prn,sr->mn", self.B, self.B, Di, optimize=True) for Di in D]
        return K

    def build_JK(self, C):
        r"""
        Compute the Coulomb and exchange matrices for a given set of orbitals.

        .. math::

            J_{\mu\nu} = \sum_{i}\sum_{\rho\sigma} (\mu\nu|\rho\sigma) C^*_{\rho i} C_{\sigma i}\\
            K_{\mu\nu} = \sum_{i}\sum_{\rho\sigma} (\mu\rho|\nu\sigma) C^*_{\rho i} C_{\sigma i}

        Parameters
        ----------
        C : list of NDArray
            List of coefficient matrices for the orbitals.
        
        Returns
        -------
        tuple(list[NDArray], list[NDArray])
            A tuple containing the lists of Coulomb (J) and exchange (K) matrices.
        """
        D = [np.einsum("mi,ni->mn", Ci, Ci.conj(), optimize=True) for Ci in C]
        J = self.build_J(D)
        K = self.build_K(C)
        return J, K

    def build_JK_generalized(self, C, g1):
        r"""
        Compute the generalized Coulomb and exchange matrices for a given set of orbitals.
        These are used in building the generalized Fock matrix in multi-reference methods.
        The generalized J and K matrices are defined as

        .. math::
            J_{\mu\nu} = \sum_{uv}\sum_{\rho\sigma} (\mu\nu|\rho\sigma) C^*_{\rho u} C_{\sigma v} \gamma_{uv}\\
            K_{\mu\nu} = \sum_{uv}\sum_{\rho\sigma} (\mu\rho|\nu\sigma) C^*_{\rho u} C_{\sigma v} \gamma_{uv}

        Parameters
        ----------
        C : NDArray
            Coefficient matrix for the orbitals.
        g1 : NDArray
            One-electron density matrix (1-RDM) in the MO basis.
        
        Returns
        -------
        tuple(NDArray, NDArray)
            A tuple containing the generalized Coulomb (J) and exchange (K) matrices.
        """
        assert C.shape[1] == g1.shape[0], "C and g1 must have compatible dimensions"

        try:
            L = np.linalg.cholesky(g1, upper=False)
            Cp = C @ L
        except np.linalg.LinAlgError:
            n, L = np.linalg.eigh(g1)
            assert np.all(n > -1.0e-11), "g1 must be positive semi-definite"
            n = np.maximum(n, 0)
            Cp = C @ L @ np.diag(np.sqrt(n))

        J, K = self.build_JK([Cp])
        return J[0], K[0]

    def two_electron_integrals_gen_block(self, C1, C2, C3, C4, antisymmetrize=False):
        r"""
        Compute the two-electron integrals for a given set of orbitals. This method is
        general and can handle different sets of orbitals for each index (p, q, r, s).

        The resulting integrals are stored in a 4D array with the following convention:
        V[p,q,r,s] = :math:`\langle pq | rs \rangle`, where

        .. math::

            \langle pq | rs \rangle = \iint \phi^*_p(r_1) \phi^*_q(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2) dr_1 dr_2


        Parameters
        ----------
        C1 : NDArray
            Coefficient matrix for the first set of orbitals (index p).
        C2 : NDArray
            Coefficient matrix for the second set of orbitals (index q).
        C3 : NDArray
            Coefficient matrix for the third set of orbitals (index r).
        C4 : NDArray
            Coefficient matrix for the fourth set of orbitals (index s).
        antisymmetrize : bool, optional, default=False
            Whether to antisymmetrize the integrals. If True, the integrals are antisymmetrized as:
            V[p,q,r,s] = :math:`\langle pq || rs \rangle = \langle pq | rs \rangle - \langle pq | sr \rangle`

        Returns
        -------
        V : NDArray
            The two-electron integrals in the form of a 4D array.
        """
        V = np.einsum(
            "Pmn,Prs,mi,rj,nk,sl->ijkl",
            self.B,
            self.B,
            C1.conj(),
            C2.conj(),
            C3,
            C4,
            optimize=True,
        )
        if antisymmetrize:
            V -= np.einsum("ijkl->ijlk", V)
        return V

    def two_electron_integrals_block(self, C, antisymmetrize=False):
        r"""
        Compute the two-electron integrals for a given set of orbitals.

        The resulting integrals are stored in a 4D array with the following convention:
        V[p,q,r,s] = :math:`\langle pq | rs \rangle`, where

        .. math::

            \langle pq | rs \rangle = \iint \phi^*_p(r_1) \phi^*_q(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2) dr_1 dr_2

        Parameters
        ----------
        C : NDArray
            Coefficient matrix for the set of orbitals.
        antisymmetrize : bool, optional, default=False
            Whether to antisymmetrize the integrals. If True, the integrals are antisymmetrized as:
            V[p,q,r,s] = :math:`\langle pq || rs \rangle = \langle pq | rs \rangle - \langle pq | sr \rangle`

        Returns
        -------
        V : NDArray
            The two-electron integrals in the form of a 4D array.
        """
        return self.two_electron_integrals_gen_block(C, C, C, C, antisymmetrize)

    def two_electron_integrals_gen_block_spinor(
        self, C1, C2, C3, C4, antisymmetrize=False
    ):
        r"""
        Compute the two-electron integrals for a given set of orbitals. This method is
        general and can handle different sets of orbitals for each index (p, q, r, s).

        The resulting integrals are stored in a 4D array with the following convention:
        V[p,q,r,s] = :math:`\langle pq | rs \rangle`, where

        .. math::

            \langle pq | rs \rangle = \iint \phi^*_p(r_1) \phi^*_q(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2) dr_1 dr_2


        Parameters
        ----------
        C1 : NDArray
            Coefficient matrix for the first set of orbitals (index p).
        C2 : NDArray
            Coefficient matrix for the second set of orbitals (index q).
        C3 : NDArray
            Coefficient matrix for the third set of orbitals (index r).
        C4 : NDArray
            Coefficient matrix for the fourth set of orbitals (index s).
        antisymmetrize : bool, optional, default=False
            Whether to antisymmetrize the integrals. If True, the integrals are antisymmetrized as:
            V[p,q,r,s] = :math:`\langle pq || rs \rangle = \langle pq | rs \rangle - \langle pq | sr \rangle`

        Returns
        -------
        V : NDArray
            The two-electron integrals in the form of a 4D array.
        """
        nbf = self.nbf
        V = np.zeros(
            (C1.shape[1], C2.shape[1], C3.shape[1], C4.shape[1]), dtype=complex
        )
        _a = slice(0, nbf)
        _b = slice(nbf, nbf * 2)
        # equivalent to 4 nested for loops over a,b parts of of C1/2/3/4
        for s1, s2, s3, s4 in itertools.product([_a, _b], repeat=4):
            V += np.einsum(
                "Pmn,Prs,mi,rj,nk,sl->ijkl",
                self.B,
                self.B,
                C1[s1, :].conj(),
                C2[s2, :].conj(),
                C3[s3, :],
                C4[s4, :],
                optimize=True,
            )

        if antisymmetrize:
            V -= np.einsum("ijkl->ijlk", V)
        return V
