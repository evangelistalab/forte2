from dataclasses import dataclass, field

import forte2
import numpy as np
import scipy as sp
import time


from forte2.jkbuilder.jkbuilder import DFFockBuilder
from forte2.helpers.mixins import MOs

from .initial_guess import minao_initial_guess


@dataclass
class RHF(MOs):
    charge: int
    econv: float = 10**-6
    dconv: float = 10**-3
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
        print(f"Energy convergence criterion: {self.econv:e}")
        print(f"Density convergence criterion: {self.dconv:e}")

        Vnn = forte2.ints.nuclear_repulsion(system.atoms)
        S = forte2.ints.overlap(system.basis)
        T = forte2.ints.kinetic(system.basis)
        V = forte2.ints.nuclear(system.basis, system.atoms)
        fock_builder = DFFockBuilder(system)

        H = T + V

        self.C = self._initial_guess(system, H, S)
        D = self._build_density_matrix(self.C)

        Eold = 0.0
        Dold = D

        diis = forte2.helpers.DIIS()

        for iter in range(self.maxiter):
            # Build the Fock matrix
            J = fock_builder.build_J([D])[0]
            K = fock_builder.build_K([self.C[:, : self.na]])[0]

            # Build the Fock matrix
            F = H + 2.0 * J - K

            # Build the DIIS error
            AO_gradient = F @ D @ S - S @ D @ F

            AO_gradient_norm = np.linalg.norm(AO_gradient, ord=np.inf)

            F = diis.update(F, AO_gradient)

            # Compute the energy
            self.E = Vnn + np.sum(D * (H + F))

            # Diagonalize the Fock matrix
            self.eps, self.C = sp.linalg.eigh(F, S)

            # Build the density matrix
            D = self._build_density_matrix(self.C)

            # check for convergence of both energy and density matrix
            print(
                f"{iter + 1:4d} {self.E:20.12f} {self.E - Eold:20.12f} {np.linalg.norm(D - Dold):20.12f} {AO_gradient_norm:20.12f}"
            )

            if (np.abs(self.E - Eold) < self.econv) and (
                np.linalg.norm(D - Dold) < self.dconv
            ):
                print("SCF iterations converged")
                break

            Eold = self.E
            Dold = D

        end = time.monotonic()
        print(f"SCF time: {end - start:.2f} seconds")

    def _build_density_matrix(self, C):
        D = np.einsum("mi,ni->mn", C[:, 0 : self.na], C[:, 0 : self.na])
        return D

    def _initial_guess(self, system, H, S):
        # Use the minao initial guess
        return minao_initial_guess(system, H, S)
