from dataclasses import dataclass, field

import forte2
import numpy as np
import scipy as sp
import time


from forte2.jkbuilder.jkbuilder import DFFockBuilder
from forte2.helpers.mixins import MOs


@dataclass
class RHF(MOs):
    charge: int
    econv: float = 1e-8
    dconv: float = 1e-4
    maxiter: int = 100

    def run(self, system):
        start = time.monotonic()
        Zsum = np.sum([x[0] for x in system.atoms])
        nel = Zsum - self.charge
        self.nbasis = system.basis.size
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
        fock_builder = DFFockBuilder(system)

        H = T + V
        D = self._initial_guess()

        k = 1.75
        # Build the GWH matrix
        GWH = np.zeros((self.nbasis, self.nbasis))
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                GWH[i, j] = 0.5 * k * (H[i, i] + H[j, j]) * S[i, j]
            GWH[i, i] = H[i, i]

        self.E = 2.0 * np.sum(D * H)

        for i in range(self.maxiter):
            Eold = self.E
            Dold = D

            # Build the Fock matrix
            J = fock_builder.build_J([D])[0]
            K = (
                fock_builder.build_K([self.C[:, : self.na]])[0]
                if i > 0
                else np.zeros((self.nbasis, self.nbasis))
            )
            F = H + 2.0 * J - K if i > 0 else GWH

            # Compute the energy
            self.E = Vnn + np.sum(D * (H + F))

            # Diagonalize the Fock matrix
            self.eps, self.C = sp.linalg.eigh(F, S)

            # Build the density matrix
            D = self._build_density_matrix(self.C)

            # check for convergence of both energy and density matrix
            print(
                f"{i + 1:4d} {self.E:20.12f} {self.E - Eold:20.12f} {np.linalg.norm(D - Dold):20.12f}"
            )
            if (np.abs(self.E - Eold) < self.econv) and (
                np.linalg.norm(D - Dold) < self.dconv
            ):
                print("SCF iterations converged")
                break

        end = time.monotonic()
        print(f"SCF time: {end - start:.2f} seconds")

    def _initial_guess(self):
        D = np.zeros((self.nbasis, self.nbasis))
        return D

    def _build_density_matrix(self, C):
        D = np.einsum("mi,ni->mn", C[:, 0 : self.na], C[:, 0 : self.na])
        return D
