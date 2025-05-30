import numpy as np
from numpy.typing import NDArray

from forte2 import ints
from forte2.jkbuilder.jkbuilder import FockBuilder


class MOIntegrals:
    """Class to compute molecular orbital integrals for a given set of orbitals.


    Args:
        C (ndarray): The coefficient matrix for the molecular orbitals.
        orbitals (list): Subspace of the orbitals for which to compute the integrals.
        core_orbitals (list, optional): Subspace of doubly occupied orbitals. Defaults to None.
    Returns:
        None
    Attributes:
        E (float): Nuclear repulsion plus the core energy contribution.
        H (ndarray): The effective one-electron integrals.
        V (ndarray): The two-electron integrals stored in physics convention: V[p,q,r,s] = <pq|rs>
    """

    def __init__(self, C: NDArray, orbitals: list, core_orbitals: list = None) -> None:
        self.C = C
        self.orbitals = orbitals
        self.core_orbitals = core_orbitals

    def run(self, system) -> None:
        jkbuilder = FockBuilder(system)
        C = self.C[:, self.orbitals]

        self.basis = system.basis
        T = ints.kinetic(system.basis, system.basis)
        V = ints.nuclear(system.basis, system.basis, system.atoms)

        # nuclear repulsion energy contribution to the energy
        self.E = ints.nuclear_repulsion(system.atoms)

        # one-electron contributions to the one-electron integrals
        self.H = np.einsum("mi,mn,nj->ij", C, T + V, C)

        if self.core_orbitals:
            # compute the J and K matrices contributions from the core orbitals
            Ccore = self.C[:, self.core_orbitals]
            J, K = jkbuilder.build_JK([Ccore])

            # one-electron contributions to the energy
            self.E += 2.0 * np.einsum("mi,mn,ni->", Ccore, T + V, Ccore)

            # two-electron contributions to the energy
            self.E += np.einsum("mi,mn,ni->", Ccore, 2 * J[0] - K[0], Ccore)

            # two-electron contributions to the one-electron integrals
            self.H += np.einsum("mi,mn,nj->ij", C, 2 * J[0] - K[0], C)

        # two-electron integrals
        self.V = jkbuilder.two_electron_integrals_block(C)
