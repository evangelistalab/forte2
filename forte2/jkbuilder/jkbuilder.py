import numpy as np
import scipy as sp
from forte2 import ints
from forte2.system import ModelSystem
from forte2.helpers import logger


class FockBuilder:
    """Class to build the Fock matrix using the Cholesky decomposition of the auxiliary basis integrals.
    This class computes the atomic Coulomb (J) and exchange (K) matrices
    using the auxiliary basis functions.

    Args:
        system (System or ModelSystem): The system for which to build the Fock matrix.
            If a ModelSystem is provided, it will decompose the 4D ERI tensor using Cholesky decomposition with complete pivoting.
        use_aux_corr (bool): Whether to use 'auxiliary_basis' or 'auxiliary_basis_corr'. Default is False.
    """

    def __init__(self, system, use_aux_corr=False):
        if isinstance(system, ModelSystem):
            # special handling for ModelSystem
            eri = system.eri
            nbf = system.nbf()
            eri = eri.reshape((nbf**2,) * 2)
            # dpstrf: Cholesky decomposition with complete pivoting
            # tol=-1 ~machine precision tolerance
            C, piv, rank, info = sp.linalg.lapack.dpstrf(eri, tol=-1)
            if info < 0:
                raise ValueError(
                    f"dpstrf failed with info={info}, indicating the {-info}-th argument had an illegal value."
                )

            piv = piv - 1  # convert to 0-based indexing
            self.B = C[:rank, piv].reshape((rank, nbf, nbf))
            print(self.B)
            system.naux = lambda: rank
            return

        self.basis = system.basis
        self.auxiliary_basis = (
            system.auxiliary_basis_corr if use_aux_corr else system.auxiliary_basis
        )

        # Compute the memory requirements
        nb = self.basis.size
        naux = self.auxiliary_basis.size
        memory_gb = 8 * (naux**2 + naux * nb**2) / (1024**3)
        logger.log_info1(f"Memory requirements: {memory_gb:.2f} GB")
        logger.log_info1(f"Number of system basis functions: {nb}")
        logger.log_info1(f"Number of auxiliary basis functions: {naux}")

        # Compute the integrals (P|Q) with P, Q in the auxiliary basis
        M = ints.coulomb_2c(self.auxiliary_basis, self.auxiliary_basis)

        # Decompose M = L L.T
        L = sp.linalg.cholesky(M)

        # Solve L.T X = I, or X = L.T^{-1} = M^{-1/2}
        I = np.eye(M.shape[0])
        M_inv_sqrt = sp.linalg.solve_triangular(L.T, I, lower=True)

        # Compute the integrals (P|mn) with P in the auxiliary basis and m, n in the system basis
        Pmn = ints.coulomb_3c(self.auxiliary_basis, system.basis, system.basis)

        # Compute B[P|mn] = M^{-1/2}[P|Q] (Q|mn)
        self.B = np.einsum("PQ,Qmn->Pmn", M_inv_sqrt, Pmn, optimize=True)

        del Pmn

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
        D = [np.einsum("mi,ni->mn", Ci, Ci.conj(), optimize=True) for Ci in C]
        J = self.build_J(D)
        K = self.build_K(C)
        return J, K

    def two_electron_integrals_gen_block(self, C1, C2, C3, C4, antisymmetrize=False):
        """Compute the two-electron integrals for a given set of orbitals. This method is
        general and can handle different sets of orbitals for each index (p, q, r, s).

        The resulting integrals are stored in a 4D array with the following convention:
            V[p,q,r,s] = <pq|rs> = ∫∫ φ*_p(r1) φ*_q(r2) (1/r12) φ_r(r1) φ_s(r2) dr1 dr2

        Args:
            C1 (ndarray): Coefficient matrix for the first set of orbitals (index p).
            C2 (ndarray): Coefficient matrix for the second set of orbitals (index q).
            C3 (ndarray): Coefficient matrix for the third set of orbitals (index r).
            C4 (ndarray): Coefficient matrix for the fourth set of orbitals (index s).
            antisymmetrize (bool): Whether to antisymmetrize the integrals.
        Returns:
            V (ndarray): The two-electron integrals in the form of a 4D array.
                If antisymmetrize is True, the integrals are antisymmetrized as:
                V[p,q,r,s] = <pq||rs> = <pq|rs> - <pq|rs>
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
        """Compute the two-electron integrals for a given set of orbitals.

        The resulting integrals are stored in a 4D array with the following convention:
            V[p,q,r,s] = <pq|rs> = ∫∫ φ*_p(r1) φ*_q(r2) (1/r12) φ_r(r1) φ_s(r2) dr1 dr2

        Args:
            C (ndarray): Coefficient matrix for the set of orbitals.
            antisymmetrize (bool): Whether to antisymmetrize the integrals.
        Returns:
            V (ndarray): The two-electron integrals in the form of a 4D array.
                If antisymmetrize is True, the integrals are antisymmetrized as:
                V[p,q,r,s] = <pq||rs> = <pq|rs> - <pq|rs>
        """
        return self.two_electron_integrals_gen_block(C, C, C, C, antisymmetrize)
