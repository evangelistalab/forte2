import numpy as np
import scipy as sp

from forte2.system import build_basis
from forte2 import ints


class DFFockBuilder:
    def __init__(self, system):
        self.basis = system.basis
        self.auxiliary_basis = system.auxiliary_basis

        # Compute the memory requirements
        nb = self.basis.size
        naux = self.auxiliary_basis.size
        memory_gb = 8 * (naux**2 + naux * nb**2) / (1024**3)
        print(f"Memory requirements: {memory_gb:.2f} GB")
        print(f"Number of system basis functions: {nb}")
        print(f"Number of auxiliary basis functions: {naux}")

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
        J = [np.einsum("Pmn,Prs,rs->mn", self.B, self.B, Di, optimize=True) for Di in D]
        return J

    def build_K(self, C, ghf=False):
        Y = [np.einsum("Pmr,ri->Pmi", self.B, Ci, optimize=True) for Ci in C]
        if ghf:
            K = []
            for Yi in Y:
                for Yj in Y:
                    K.append(np.einsum("Pmi,Pni->mn", Yi.conj(), Yj, optimize=True))
        else:
            K = [np.einsum("Pmi,Pni->mn", Yi.conj(), Yi, optimize=True) for Yi in Y]
        return K
    
    def build_K_density(self, D):
        K = [np.einsum("Pms,Prn,rs->mn", self.B, self.B, Di, optimize=True) for Di in D]
        return K


    def build_JK(self, C):
        D = [np.einsum("mi,ni->mn", Ci, Ci, optimize=True) for Ci in C]
        J = self.build_J(D)
        K = self.build_K(C)
        return J, K

    def two_electron_integrals_gen_block(self, C1, C2, C3, C4, antisymmetrize=False):
        V = np.einsum(
            "Pmn,Prs,mi,rj,nk,sl->ijkl",
            self.B,
            self.B,
            C1,
            C2,
            C3,
            C4,
            optimize=True,
        )
        if antisymmetrize:
            V -= np.einsum("ijkl->ijlk", V)
        return V

    def two_electron_integrals_block(self, C, antisymmetrize=False):
        return self.two_electron_integrals_gen_block(C, C, C, C, antisymmetrize)
