from dataclasses import dataclass, field

import forte2
import numpy as np
import scipy as sp
import time


from forte2.jkbuilder.jkbuilder import DFFockBuilder
from forte2.helpers.mixins import MOs

from .initial_guess import minao_initial_guess


@dataclass
class UHF(MOs):
    charge: int
    mult: int
    econv: float = 1e-8
    dconv: float = 1e-4
    maxiter: int = 100

    def run(self, system):
        start = time.monotonic()
        Zsum = np.sum([x[0] for x in system.atoms])
        nel = Zsum - self.charge
        self.nbasis = system.basis.size
        self.na = (nel + self.mult - 1) // 2 
        self.nb = (nel - self.mult + 1) // 2
        self.sz = (self.na - self.nb) / 2.0

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

        self.C = self._initial_guess(system, H, S)
        D = self._build_density_matrix()

        # Introduce spin polarization in initial guess for non-singlets
        if self.mult != 1:
            D[1] *= 0.0

        Eold = 0.0
        Dold = D

        diis = forte2.helpers.DIIS()
        
        self.eps = [None for _ in range(2)]
        for i in range(self.maxiter):

            # Build the Fock matrix
            Ja, Jb = fock_builder.build_J(D)
            K = fock_builder.build_K([self.C[0][:, :self.na], self.C[1][:, :self.nb]])
            F = [H + Ja + Jb - k for k in K]

            # Build the DIIS error
            AO_gradient = np.hstack([(f @ d @ S - S @ d @ f).flatten() for d, f in zip(D, F)])

            AO_gradient_norm = np.linalg.norm(AO_gradient, ord=np.inf)

            F_flat = diis.update(np.hstack([f.flatten() for f in F]), AO_gradient)
            F = [F_flat[:self.nbasis**2].reshape(self.nbasis, self.nbasis), F_flat[self.nbasis**2:].reshape(self.nbasis, self.nbasis)]

            # Compute the energy
            self.E = Vnn + self._energy(D, H, F) 

            # Diagonalize the Fock matrix
            self.eps[0], self.C[0] = sp.linalg.eigh(F[0], S)
            self.eps[1], self.C[1] = sp.linalg.eigh(F[1], S)

            # Build the density matrix
            D = self._build_density_matrix()

            # Change in density & energy
            ddm = np.linalg.norm(D[0] - Dold[0]) + np.linalg.norm(D[1] - Dold[1])

            # Compute spin <S^2> value
            self.s2 = self._spin(S)

            # check for convergence of both energy and density matrix
            print(
                    f"{i + 1:4d} {self.E:20.12f} {self.E - Eold:20.12f} {ddm:20.12f} {self.s2:20.12f}"
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

    def _spin(self, S):
        # alpha-beta orbital overlap matrix 
        # S_ij = < psi_i | psi_j >, i,j=occ
        #      = \sum_{uv} c_ui^* c_vj <u|v>
        S_ij = np.einsum("ui,uv,vj->ij", self.C[0][:, :self.na].conj(), S, self.C[1][:, :self.nb], optimize=True)
        # Spin contamination: <s^2> - <s^2>_exact = N_b - \sum_{ij} |S_ij|^2
        ds2 = self.nb - np.einsum("ij,ij->", S_ij.conj(), S_ij)
        # <S^2> value
        s2 = self.sz*(self.sz + 1) + ds2
        return s2

    def _initial_guess(self, system, H, S):
        # Use the minao initial guess
        C = minao_initial_guess(system, H, S)
        return [C, C]

    def _build_density_matrix(self):
        D_a = np.einsum("mi,ni->mn", self.C[0][:, :self.na], self.C[0][:, :self.na])
        D_b = np.einsum("mi,ni->mn", self.C[1][:, :self.nb], self.C[1][:, :self.nb])
        return [D_a, D_b]

    def _energy(self, D, H, F):
        energy = 0.5 * (
                np.einsum("vu,uv->", D[0] + D[1], H)
                + np.einsum("vu,uv->", D[0], F[0])
                + np.einsum("vu,uv->", D[1], F[1])
        )
        return energy

