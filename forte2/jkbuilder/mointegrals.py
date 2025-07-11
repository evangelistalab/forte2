from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from forte2 import ints
from forte2.jkbuilder.jkbuilder import FockBuilder
from forte2.system.system import System


@dataclass
class RestrictedMOIntegrals:
    """Class to compute molecular orbital integrals for a given set of restricted orbitals.

    Args:
        C (ndarray): The coefficient matrix for the molecular orbitals.
        orbitals (list): Subspace of the orbitals for which to compute the integrals.
        core_orbitals (list, optional): Subspace of doubly occupied orbitals. Defaults to None.
        use_aux_corr (bool, optional): If True, use 'auxiliary_basis_corr', else use 'auxiliary_basis'. Defaults to False
    Returns:
        None
    Attributes:
        E (float): Nuclear repulsion plus the core energy contribution.
        H (ndarray): The effective one-electron integrals.
        V (ndarray): The two-electron integrals stored in physics convention: V[p,q,r,s] = <pq|rs>
    """

    system: System
    C: NDArray
    orbitals: list
    core_orbitals: list = field(default_factory=list)
    use_aux_corr: bool = False

    def __post_init__(self):
        jkbuilder = FockBuilder(self.system, self.use_aux_corr)
        C = self.C[:, self.orbitals]

        basis = self.system.basis
        atoms = self.system.atoms
        T = ints.kinetic(basis)
        V = ints.nuclear(basis, atoms)

        # nuclear repulsion energy contribution to the energy
        self.E = ints.nuclear_repulsion(atoms)

        # one-electron contributions to the one-electron integrals
        self.H = np.einsum("mi,mn,nj->ij", C, T + V, C)

        if self.core_orbitals:
            # compute the J and K matrices contributions from the core orbitals
            Ccore = self.C[:, self.core_orbitals]

            # one-electron contributions to the energy
            self.E += 2.0 * np.einsum("mi,mn,ni->", Ccore, T + V, Ccore)

            J, K = jkbuilder.build_JK([Ccore])

            # two-electron contributions to the energy
            self.E += np.einsum("mi,mn,ni->", Ccore, 2 * J[0] - K[0], Ccore)

            # two-electron contributions to the one-electron integrals
            self.H += np.einsum("mi,mn,nj->ij", C, 2 * J[0] - K[0], C)

        # two-electron integrals
        self.V = jkbuilder.two_electron_integrals_block(C)
