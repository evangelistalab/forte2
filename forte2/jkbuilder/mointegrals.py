from dataclasses import dataclass, field

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from forte2.jkbuilder.jkbuilder import FockBuilder
from forte2.system.system import System

# module-level tokens
o, v = object(), object()

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
    fock_builder: FockBuilder = None
    spinorbital: bool = False

    def __post_init__(self):
        if self.fock_builder is None:
            jkbuilder = FockBuilder(self.system, self.use_aux_corr)
        else:
            jkbuilder = self.fock_builder
        C = self.C[:, self.orbitals]

        # nuclear repulsion energy contribution to the energy
        self.E = self.system.nuclear_repulsion

        # build one-electron integrals
        H_ao = self.system.ints_hcore()

        # one-electron contributions to the one-electron integrals
        self.H = np.einsum("mi,mn,nj->ij", C, H_ao, C)

        if self.core_orbitals:
            # compute the J and K matrices contributions from the core orbitals
            Ccore = self.C[:, self.core_orbitals]

            # one-electron contributions to the energy
            self.E += 2.0 * np.einsum("mi,mn,ni->", Ccore, H_ao, Ccore)

            J, K = jkbuilder.build_JK([Ccore])

            # two-electron contributions to the energy
            self.E += np.einsum("mi,mn,ni->", Ccore, 2 * J[0] - K[0], Ccore)

            # two-electron contributions to the one-electron integrals
            self.H += np.einsum("mi,mn,nj->ij", C, 2 * J[0] - K[0], C)

        # two-electron integrals
        self.V = jkbuilder.two_electron_integrals_block(C)

    def convert_to_spinorbital(self):
        '''
        Convert restricted spatial orbitals into spinorbitals.
        '''
        nso = 2 * self.system.nbf

        temp = self.H.copy()
        self.H = np.zeros((nso, nso))
        self.H[::2, ::2] = temp 
        self.H[1::2, 1::2] = temp

        temp = self.V.copy()
        self.V = np.zeros((nso, nso, nso, nso))
        self.V[::2, ::2, ::2, ::2] = temp - temp.transpose(0, 1, 3,2) # v(aa)
        self.V[1::2, 1::2, 1::2, 1::2] = temp - temp.transpose(0, 1, 3,2) # v(bb)
        self.V[::2, 1::2, ::2, 1::2] = temp
        self.V[1::2, ::2, 1::2, ::2] = temp.transpose(1, 0, 3, 2)
        self.V[::2, 1::2, 1::2, ::2] = -temp.transpose(0, 1, 3, 2)
        self.V[1::2, ::2, ::2, 1::2] = -temp.transpose(1, 0, 2, 3)

        self.spinorbital = True

@dataclass
class SRRestrictedMOIntegrals:
    r"""
    Class to compute molecular orbital integrals for a given set of restricted orbitals
    for use in SR correlation methods.

    Parameters
    ----------
    moints : RestrictedMOIntegrals
        The integrals in the MO basis
    frozen : int, optional
        Number of frozen core orbitals to be excluded from the correlated calculation. Defaults to 0.
    virtual : int, optional
        Number of virtual orbitals to be excluded from the correlated calculation. Defaults to 0.


    Attributes
    ----------
    """

    ints: RestrictedMOIntegrals
    frozen: int = 0
    virtual: int = 0

    def __post_init__(self):
        if not self.ints.spinorbital:
            raise ValueError("Expected restricted spin-orbital integrals (ints.spinorbital == True).")
        
        self.o = slice(0, self.ints.system.nel - self.frozen)
        self.v = slice(self.ints.system.nel, self.ints.system.nbf - self.virtual)

        self.build_fock()

    def __getitem__(self, key):

        idx = tuple(self.o if k == 'o' else self.v if k == 'v' else k for k in key)
        if len(idx) == 2:
            return self.F[idx]
        if len(idx) == 4:
            return self.ints.V[idx]
        raise IndexError("Use 2 indices for H or 4 indices for V.")
    
    def build_fock(self):
        self.F = self.ints.H + np.einsum("piqi->pq", self.ints.V[:, self.o, :, self.o])
        self.eps_o = np.diag(self.F[self.o, self.o])
        self.eps_v = np.diag(self.F[self.v, self.v])

    def scf_energy(self):
        E = (
            np.einsum('ii->', self.ints.H[self.o, self.o])
            + 0.5 * np.einsum("ijij->", self.ints.V[self.o, self.o, self.o, self.o])
        )
        E += self.ints.system.nuclear_repulsion
        return E
    
    def scf_energy_fock(self):
        E = (
            np.einsum('ii->', self.F[self.o, self.o])
            - 0.5 * np.einsum("ijij->", self.ints.V[self.o, self.o, self.o, self.o])
        )
        E += self.ints.system.nuclear_repulsion
        return E