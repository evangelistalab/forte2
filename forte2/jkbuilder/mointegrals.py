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
    fock_builder : FockBuilder, optional
        An instance of FockBuilder to use for building the Fock matrix.
        If not provided, a new FockBuilder will be created.
    antisymmetrize : bool, optional, default=False
        If True, antisymmetrize the two-electron integrals.
    spinorbital : bool, optional, default=False
        If True, the integrals are converted to the spin-orbital basis.

    Attributes
    ----------
    E : float
        Nuclear repulsion plus the core energy contribution.
    H : NDArray
        The effective one-electron integrals.
    V : NDArray
        The two-electron integrals stored in physicist's convention: V[p,q,r,s] = :math:`\langle pq | rs \rangle`.
    """

    system: System
    C: NDArray
    orbitals: list
    core_orbitals: list = field(default_factory=list)
    use_aux_corr: bool = False
    fock_builder: FockBuilder = None
    antisymmetrize: bool = False
    spinorbital: bool = False

    def __post_init__(self):
        self.norb = len(self.orbitals)
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
        self.V = jkbuilder.two_electron_integrals_block(
            C, antisymmetrize=self.antisymmetrize
        )

        if self.spinorbital:
            self._convert_to_spinorbital()

    def _convert_to_spinorbital(self):
        """
        Convert restricted spatial orbitals into spinorbitals.
        """
        nso = 2 * self.norb

        temp = self.H.copy()
        self.H = np.zeros((nso, nso))
        self.H[::2, ::2] = temp
        self.H[1::2, 1::2] = temp

        temp = self.V.copy()
        self.V = np.zeros((nso, nso, nso, nso))
        if self.antisymmetrize:
            self.V[::2, ::2, ::2, ::2] = temp - temp.transpose(0, 1, 3, 2)  # v(aa)
            self.V[1::2, 1::2, 1::2, 1::2] = temp - temp.transpose(0, 1, 3, 2)  # v(bb)
            self.V[::2, 1::2, ::2, 1::2] = temp
            self.V[1::2, ::2, 1::2, ::2] = temp.transpose(1, 0, 3, 2)
            self.V[::2, 1::2, 1::2, ::2] = -temp.transpose(0, 1, 3, 2)
            self.V[1::2, ::2, ::2, 1::2] = -temp.transpose(1, 0, 2, 3)
        else:
            self.V[::2, ::2, ::2, ::2] = temp
            self.V[1::2, 1::2, 1::2, 1::2] = temp
            self.V[::2, 1::2, ::2, 1::2] = temp
            self.V[1::2, ::2, 1::2, ::2] = temp.transpose(1, 0, 3, 2)


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

    def __init__(self, scf, frozen=0, virtual=0, spinorbital=True):

        self.ints = RestrictedMOIntegrals(scf.system, 
                                          scf.C[0],
                                          orbitals=list(range(scf.nmo)), 
                                          spinorbital=spinorbital, 
                                          antisymmetrize=True)
        
        self.occupied_all = slice(0, scf.nel)
        self.unoccupied_all = slice(scf.nel, 2 * self.ints.norb)
        self.o = slice(frozen, scf.nel)
        self.v = slice(scf.nel, 2 * self.ints.norb - virtual) 
        self.E = self.scf_energy()

        self.build_fock()

        assert np.allclose(self.E, self.scf_energy_fock())


    def __getitem__(self, key):
        idx = tuple(self.o if k == "o" else self.v if k == "v" else k for k in key)
        if len(idx) == 2:
            return self.F[idx]
        if len(idx) == 4:
            return self.ints.V[idx]
        raise IndexError("Use 2 indices for H or 4 indices for V.")

    def build_fock(self):
        self.F = self.ints.H + np.einsum("piqi->pq", self.ints.V[:, self.occupied_all, :, self.occupied_all])

    def scf_energy(self):
        E = np.einsum("ii->", self.ints.H[self.occupied_all, self.occupied_all]) + 0.5 * np.einsum("ijij->", self.ints.V[self.occupied_all, self.occupied_all, self.occupied_all, self.occupied_all])
        E += self.ints.system.nuclear_repulsion
        return E

    def scf_energy_fock(self):
        E = np.einsum("ii->", self.F[self.occupied_all, self.occupied_all]) - 0.5 * np.einsum(
            "ijij->", self.ints.V[self.occupied_all, self.occupied_all, self.occupied_all, self.occupied_all]
        )
        E += self.ints.system.nuclear_repulsion
        return E
