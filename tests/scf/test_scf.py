import forte2
import numpy as np
import scipy as sp
import time


class DFFockBuilder:
    def __init__(self, system, auxiliary_basis):
        self.basis = system.basis
        self.auxiliary_basis = forte2.system.build_basis(auxiliary_basis, system.atoms)

        # Compute the memory requirements
        nb = self.basis.size
        naux = self.auxiliary_basis.size
        memory_gb = 8 * (naux**2 + naux * nb**2) / (1024**3)
        print(f"Memory requirements: {memory_gb:.2f} GB")
        print(f"Number of system basis functions: {nb}")
        print(f"Number of auxiliary basis functions: {naux}")

        # Compute the integrals (P|Q) with P, Q in the auxiliary basis
        M = forte2.ints.coulomb_2c(self.auxiliary_basis, self.auxiliary_basis)

        # Decompose M = L L.T
        L = sp.linalg.cholesky(M)

        # Solve L.T X = I, or X = L.T^{-1} = M^{-1/2}
        I = np.eye(M.shape[0])
        M_inv_sqrt = sp.linalg.solve_triangular(L.T, I, lower=True)

        # Compute the integrals (P|mn) with P in the auxiliary basis and m, n in the system basis
        Pmn = forte2.ints.coulomb_3c(self.auxiliary_basis, system.basis, system.basis)

        # Compute B[P|mn] = M^{-1/2}[P|Q] (Q|mn)
        self.B = np.einsum("PQ,Qmn->Pmn", M_inv_sqrt, Pmn, optimize=True)

        del Pmn

    def build_J(self, D):
        J = [np.einsum("Pmn,Prs,rs->mn", self.B, self.B, Di, optimize=True) for Di in D]
        return J

    def build_K(self, C):
        Y = [np.einsum("Pmr,ri->Pmi", self.B, Ci, optimize=True) for Ci in C]
        K = [np.einsum("Pmi,Pni->mn", Yi, Yi, optimize=True) for Yi in Y]
        return K


class RHF:
    def __init__(
        self, charge, econv=1e-8, dconv=1e-4, maxiter=100, auxiliary_basis=None
    ):
        self.charge = charge
        self.maxiter = maxiter
        self.econv = econv
        self.dconv = dconv
        self.auxiliary_basis = auxiliary_basis

    def run(self, system):
        start = time.monotonic()
        self.nbasis = system.basis.size
        Zsum = np.sum([x[0] for x in system.atoms])
        nel = Zsum - self.charge
        self.na = nel // 2
        self.nb = nel // 2

        print(f"Number of alpha electrons: {self.na}")
        print(f"Number of beta electrons: {self.nb}")
        print(f"Total charge: {self.charge}")

        print(f"Number of electrons: {nel}")
        print(f"Number of basis functions: {self.nbasis}")

        Vnn = forte2.ints.nuclear_repulsion(system.atoms)
        S = forte2.ints.overlap(system.basis)
        T = forte2.ints.kinetic(system.basis)
        V = forte2.ints.nuclear(system.basis, system.atoms)
        fock_builder = DFFockBuilder(system, self.auxiliary_basis)

        H = T + V
        D = self._initial_guess()

        k = 1.75
        # Build the GWH matrix
        GWH = np.zeros((self.nbasis, self.nbasis))
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                GWH[i, j] = 0.5 * k * (H[i, i] + H[j, j]) * S[i, j]
            GWH[i, i] = H[i, i]

        E = 2.0 * np.sum(D * H)

        for i in range(self.maxiter):
            Eold = E
            Dold = D

            # Build the Fock matrix
            J = fock_builder.build_J([D])[0]
            K = (
                fock_builder.build_K([C[:, : self.na]])[0]
                if i > 0
                else np.zeros((self.nbasis, self.nbasis))
            )
            F = H + 2.0 * J - K if i > 0 else GWH

            # Compute the energy
            E = Vnn + np.sum(D * (H + F))

            # Diagonalize the Fock matrix
            eps, C = sp.linalg.eigh(F, S)

            # Build the density matrix
            D = self._build_density_matrix(C)

            # check for convergence of both energy and density matrix
            print(
                f"{i + 1:4d} {E:20.12f} {E - Eold:20.12f} {np.linalg.norm(D - Dold):20.12f}"
            )
            if (np.abs(E - Eold) < self.econv) and (
                np.linalg.norm(D - Dold) < self.dconv
            ):
                print("SCF iterations converged")
                break

        self.E = E
        self.C = C

        end = time.monotonic()
        print(f"SCF time: {end - start:.2f} seconds")

    def _initial_guess(self):
        D = np.zeros((self.nbasis, self.nbasis))
        return D

    def _build_density_matrix(self, C):
        D = np.einsum("mi,ni->mn", C[:, 0 : self.na], C[:, 0 : self.na])
        return D


xyz = """
O            0.000000000000     0.000000000000    -0.061664597388
H            0.000000000000    -0.711620616369     0.489330954643
H            0.000000000000     0.711620616369     0.489330954643
"""

system = forte2.System(xyz=xyz, basis="cc-pVQZ")

scf = RHF(charge=0, auxiliary_basis="cc-pVQZ-JKFIT")
scf.run(system)
assert np.isclose(
    scf.E, -76.0614664043887672, atol=1e-10
), f"SCF energy {scf.E} is not close to expected value -76.0614664043887672"
