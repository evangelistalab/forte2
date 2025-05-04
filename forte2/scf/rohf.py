from dataclasses import dataclass, field

import forte2
import numpy as np
import scipy as sp
import time


from forte2.jkbuilder.jkbuilder import DFFockBuilder
from forte2.helpers.mixins import MOs

from .initial_guess import minao_initial_guess


# Remember, default ROHF canonicalization is different in different codes
# GAMESS - Roothaan
# PySCF - Roothaan
# NWChem - Guest-Saunders
# Psi4 - Guest-Saunders(?)
_A_ROHF = {"Davidson": [0.5, 1, 1],
           "Roothaan": [-0.5, 0.5, 1.5],
           "Guest-Saunders": [0.5, 0.5, 0.5]}
_B_ROHF = {"Davidson": [0.5, 0, 0],
           "Roothaan": [1.5, 0.5, -0.5],
           "Guest-Saunders": [0.5, 0.5, 0.5]}

@dataclass
class ROHF(MOs):
    charge: int
    mult: int
    econv: float = 1e-8
    dconv: float = 1e-4
    maxiter: int = 100
    canonicalization: str = "Roothaan"

    def run(self, system):
        start = time.monotonic()
        Zsum = np.sum([x[0] for x in system.atoms])
        nel = Zsum - self.charge
        self.nbasis = system.basis.size
        self.na = (nel + self.mult - 1) // 2 
        self.nb = (nel - self.mult + 1) // 2

        # Get canonicalization scheme
        # Note: We are sticking with Roothaan for now..
        # self.A = _A_ROHF[self.canonicalization]
        # self.B = _B_ROHF[self.canonicalization]

        print(f"Number of alpha electrons: {self.na}")
        print(f"Number of beta electrons: {self.nb}")
        print(f"Total charge: {self.charge}")
        print(f"Spin multiplicity: {self.mult}")

        print(f"Number of electrons: {nel}")
        print(f"Number of basis functions: {self.nbasis}")

        Vnn = forte2.ints.nuclear_repulsion(system.atoms)
        S = forte2.ints.overlap(system.basis)
        T = forte2.ints.kinetic(system.basis)
        V = forte2.ints.nuclear(system.basis, system.atoms)
        fock_builder = DFFockBuilder(system)
        H = T + V

        # Initial guess (Hcore, for now)
        self.C = self._initial_guess(system, H, S)
        D = self._build_density_matrix()

        Eold = 0.0
        Dold = D

        diis = forte2.helpers.DIIS()

        for i in range(self.maxiter):

            # Build the Fock matrix
            Ja, Jb = fock_builder.build_J(D)
            K = fock_builder.build_K([self.C[:, :self.na], self.C[:, :self.nb]])
            F = [H + Ja + Jb - k for k in K]

            # Build ROHF fock matrix
            F_canon = self._build_fock(D, H, F, S)

            # Build the DIIS error
            Deff = 0.5 * (D[0] + D[1])
            AO_gradient = F_canon @ Deff @ S - S @ Deff @ F_canon
            AO_gradient_norm = np.linalg.norm(AO_gradient, ord=np.inf)
            F_canon = diis.update(F_canon, AO_gradient)

            # Diagonalize the Fock matrix
            self.eps, self.C = sp.linalg.eigh(F_canon, S)

            # Build the density matrix
            D = self._build_density_matrix()

            # Compute the energy
            self.E = Vnn + self._energy(D, H, F) 

            # Change in density & energy
            ddm = np.linalg.norm(D[0] - Dold[0]) + np.linalg.norm(D[1] - Dold[1])

            # check for convergence of both energy and density matrix
            print(
                f"{i + 1:4d} {self.E:20.12f} {self.E - Eold:20.12f} {ddm:20.12f}"
            )
            if (np.abs(self.E - Eold) < self.econv) and (
                ddm < self.dconv
            ):
                print("SCF iterations converged")
                break

            Eold = self.E
            Dold = D 


        end = time.monotonic()
        print(f"SCF time: {end - start:.2f} seconds")


    def _build_fock(self, D, H, F, S):

        # Projector for core, open-shell, and virtual
        U_core = np.dot(D[1], S)
        U_open = np.dot(D[0] - D[1], S)
        U_virt = np.eye(self.nbasis) - np.dot(D[0], S)

        # Closed-shell Fock
        F_cs = 0.5 * (F[0] + F[1])

        def _project(u, v, f):
            return np.einsum("ur,uv,vt->rt", u, f, v, optimize=True)

        # these are scaled by 0.5 to account for fock + fock.T below
        fock = _project(U_core, U_core, F_cs) * 0.5
        fock += _project(U_open, U_open, F_cs) * 0.5
        fock += _project(U_virt, U_virt, F_cs) * 0.5
        # off-diagonal blocks
        fock += _project(U_open, U_core, F[1])
        fock += _project(U_open, U_virt, F[0])
        fock += _project(U_virt, U_core, F_cs)
        fock = fock + fock.conj().T
        return fock

    def _initial_guess(self, system, H, S):
        # Use the minao initial guess
        return minao_initial_guess(system, H, S)

    def _build_density_matrix(self):
        D_a = np.einsum("mi,ni->mn", self.C[:, :self.na], self.C[:, :self.na])
        D_b = np.einsum("mi,ni->mn", self.C[:, :self.nb], self.C[:, :self.nb])
        return [D_a, D_b]

    def _energy(self, D, H, F):
        energy = 0.5 * (
                np.einsum("vu,uv->", D[0] + D[1], H)
                + np.einsum("vu,uv->", D[0], F[0])
                + np.einsum("vu,uv->", D[1], F[1])
        )
        return energy
