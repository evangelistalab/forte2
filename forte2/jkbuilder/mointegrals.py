from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from forte2 import ints
from forte2.jkbuilder.jkbuilder import FockBuilder
from forte2.system.system import System


@dataclass
class RestrictedMOIntegrals:
    r"""
    Class to compute molecular orbital integrals for a given set of restricted orbitals.

    Parameters
    ----------
        C : NDArray
            The coefficient matrix for the molecular orbitals.
        orbitals : list[int]
            Subspace of the orbitals for which to compute the integrals.
        core_orbitals : list[int], optional
            Subspace of doubly occupied orbitals. Defaults to None.
        use_aux_corr : bool, optional, default=False
            If True, use ``system.auxiliary_basis_set_corr``, else use ``system.auxiliary_basis``.

    Attributes
    ----------
        E : float
            Nuclear repulsion plus the core energy contribution.
        H : NDArray
            The effective one-electron integrals.
        V : NDArray
            The two-electron integrals stored in physics convention: V[p,q,r,s] = :math:`\langle pq | rs \rangle`.
    """

    system: System
    C: NDArray
    orbitals: list
    core_orbitals: list = field(default_factory=list)
    use_aux_corr: bool = False

    def __post_init__(self):
        jkbuilder = FockBuilder(self.system, self.use_aux_corr)
        C = self.C[:, self.orbitals]

        # nuclear repulsion energy contribution to the energy
        self.E = self.system.nuclear_repulsion

        # one-electron contributions to the one-electron integrals
        self.H = np.einsum("mi,mn,nj->ij", C, self.system.ints_hcore(), C)

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
