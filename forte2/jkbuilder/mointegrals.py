import time
from dataclasses import dataclass, field

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from forte2.jkbuilder.jkbuilder import FockBuilder
from forte2.system.system import System
from forte2.helpers import logger
from forte2.helpers.comparisons import approx_vtight


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


class SONormalOrderedIntegrals:
    r"""
    Class to compute molecular spinorbital integrals for a given set of restricted (RHF/ROHF/GHF) orbitals
    for use in SR correlation methods.

    Parameters
    ----------
    scf : 
        The mean-field RHF/ROHF/GHF object.
    frozen : int, optional
        Number of frozen core spinorbitals to be excluded from the correlated calculation. Defaults to 0.
    virtual : int, optional
        Number of virtual spinorbitals to be excluded from the correlated calculation. Defaults to 0.
    build_nvirt : int, optional
        Builds integral blocks containing up to `build_nvirt` virtual dimensions. Defaults to 4.

    Attributes
    ----------
    F : dict
        The one-electron Fock integrals in the molecular spinorbital basis stored as occupied/virtual blocks in a dictionary
    V : dict
        The two-electron integrals in the molecular spinorbital basis stored as occupied/virtual blocks in a dictionary.
    B : dict
        The density-fitted tensor B(x|pq) stored as occupied/virtual slices in a dictionary
    """

    def __init__(self, scf, frozen=0, virtual=0, build_nvirt=4):

        assert scf.method in ['RHF', 'ROHF', 'GHF'], "SONormalOrderedIntegrals only supports RHF, ROHF, and GHF references."

        ### Set up orbital dimensions
        self.nocc = scf.nel                               # total number of occupied spinorbitals
        self.norb = 2 * scf.C[0].shape[1]                 # total number of orbitals
        self.na = scf.na                                  # number of alpha electrons
        self.nb = scf.nb                                  # number of beta electrons
        self.occupied_all = slice(0, self.nocc)           # slice of occupied orbitals
        self.unoccupied_all = slice(self.nocc, self.norb) # slice of unoccupied orbitals
        self.o = slice(frozen, self.nocc)                 # slice of correlated occupied orbitals
        self.v = slice(self.nocc, self.norb - virtual)    # slice of correlated unoccupied orbitals

        ### Build the integral data
        self.build_integral_data(scf, build_nvirt)

    def __getitem__(self, key):
        if len(key) == 4:
            return self.V[key]
        elif len(key) == 2:
            return self.F[key]

    def build_integral_data(self, scf, build_nvirt):

        if scf.method == 'RHF' or scf.method == 'GHF':
            self.mo_occ = [2.0] * scf.ndocc + [0.0] * scf.nuocc
        if scf.method == 'ROHF':
            self.mo_occ = [2.0] * scf.ndocc + [1.0] * scf.nsocc + [0.0] * scf.nuocc

        # Store nuclear repulsion energy
        self.nuclear_repulsion = scf.system.nuclear_repulsion
        # Obtain reordering array for singly occupied orbitals
        if scf.method == 'ROHF':
            # ab interleaving is not compatible with singly occupied alpha orbitals
            perm = self.reorder_occ_first()
        # DF tensor
        tic = time.time()
        B = self.get_df_tensor(scf.system, scf.C[0])
        if scf.method == 'ROHF':
            B = B[np.ix_(np.arange(B.shape[0]), perm, perm)]
        logger.log_debug(f"[SOIntegrals] AO-to-MO + spinorbital transformation of DF tensor: {time.time() - tic} seconds")
        # onebody (Hcore) integrals
        tic = time.time()
        Z = self.get_onebody_ints(scf.system, scf.C[0])
        if scf.method == 'ROHF':
            Z = Z[np.ix_(perm, perm)]
        logger.log_debug(f"[SOIntegrals] Building one-body spinorbital integrals: {time.time() - tic} seconds")
        # Build Fock matrix using B tensors
        tic = time.time()
        F = self.build_fock(Z, B)
        logger.log_debug(f"[SOIntegrals] Building spinorbital Fock matrix: {time.time() - tic} seconds")
        # Compute SCF energy
        self.E = self.scf_energy(Z, B)
        # Store B tensor in correlated o/v slices
        self.B = {'oo': B[:, self.o, self.o], 
                  'vv': B[:, self.v, self.v],
                  'ov': B[:, self.o, self.v],
                  'vo': B[:, self.v, self.o]}
        # Store F matrix in correlated o/v slices
        self.F = {'oo': F[self.o, self.o],
                  'vv': F[self.v, self.v],
                  'ov': F[self.o, self.v],
                  'vo': F[self.v, self.o]}
        # Compute blocks of two-electron integrals
        tic = time.time()
        self.get_twobody_ints(build_nvirt)
        logger.log_debug(f"[SOIntegrals] Building blocks (up to nvirt = {build_nvirt}) of two-electron spinorbital integrals: {time.time() - tic} seconds")
        assert self.E == approx_vtight(self.scf_energy_fock(F, B))
        del B, F, Z

    def get_df_tensor(self, system, C):
         
        jkbuilder = FockBuilder(system, use_aux_corr=False)

        # Build spinorbital B tensor in the MO basis
        B_mo = np.einsum("xij,ip,jq->xpq", jkbuilder.B, C, C)
        B = np.zeros((B_mo.shape[0], self.norb, self.norb))
        B[:, ::2, ::2] = B[:, 1::2, 1::2] = B_mo
        return B

    def get_onebody_ints(self, system, C):

        # Build spinorbital one-electron integrals in the MO basis
        Z_mo = np.einsum("mi,mn,nj->ij", C, system.ints_hcore(), C)
        Z = np.zeros((self.norb, self.norb))
        Z[::2, ::2] = Z[1::2, 1::2] = Z_mo
        return Z
    
    def get_twobody_ints(self, build_nvirt):

        # unique two-electron integrals by o-v blocks
        blk_0v = ['oooo']
        blk_1v = ['ooov', 'vooo', ]
        blk_2v = ['oovv', 'vvoo', 'voov']
        blk_3v = ['vvov', 'vovv']
        blk_4v = ['vvvv']
        integral_blocks = [blk_0v, blk_1v, blk_2v, blk_3v, blk_4v]

        # two-electron integrals via DF tensors
        self.V = {}
        for block in integral_blocks[:build_nvirt + 1]:
            for key in block:
                s1 = key[0] + key[2] # pr
                s2 = key[1] + key[3] # qs
                s3 = key[0] + key[3] # ps
                s4 = key[1] + key[2] # qr
                self.V[key] = (
                              np.einsum('xpr,xqs->pqrs', self.B[s1], self.B[s2], optimize=True) # B(pr)*B(qs) = <pq|rs>
                            - np.einsum('xps,xqr->pqrs', self.B[s3], self.B[s4], optimize=True) # B(ps)*B(qr) = <pq|sr>
                )

    def build_fock(self, Z, B):
        F = Z + (
              np.einsum("xpq,xii->pq", B, B[:, self.occupied_all, self.occupied_all], optimize=True) # <pi|v|qi>
            - np.einsum("xpi,xiq->pq", B[:, :, self.occupied_all], B[:, self.occupied_all, :], optimize=True) # <pi|v|iq>
        )
        return F

    def scf_energy(self, Z, B):
        E = (
                np.einsum("ii->", Z[self.occupied_all, self.occupied_all])
                + 0.5 * np.einsum("xii,xjj->", B[:, self.occupied_all, self.occupied_all], B[:, self.occupied_all, self.occupied_all]) 
                - 0.5 * np.einsum("xij,xji->", B[:, self.occupied_all, self.occupied_all], B[:, self.occupied_all, self.occupied_all])
                + self.nuclear_repulsion
        )
        return E

    def scf_energy_fock(self, F, B):
        E = (
                np.einsum("ii->", F[self.occupied_all, self.occupied_all]) 
                - 0.5 * np.einsum("xii,xjj->", B[:, self.occupied_all, self.occupied_all], B[:, self.occupied_all, self.occupied_all]) 
                + 0.5 * np.einsum("xij,xji->", B[:, self.occupied_all, self.occupied_all], B[:, self.occupied_all, self.occupied_all])
                + self.nuclear_repulsion
        )
        return E

    def reorder_occ_first(self):
        """
        Reorder spin–orbital integrals so that occupied spin–orbitals are a
        contiguous block. Works for RHF and ROHF (single set of spatial MOs).
        """
        # Decide which spin is occupied for singly-occupied spatial MOs.
        singles_spin = 0 if self.na >= self.nb else 1      # 0: alpha, 1: beta

        # Build occupied spin–orbital indices in the current (αβ interleaved) order.
        occ_idx = []
        for p, occ in enumerate(self.mo_occ):
            if occ > 1.5:                         # doubly occupied
                occ_idx.extend([2*p, 2*p+1])      # α and β
            elif occ > 0.5:                       # singly occupied
                occ_idx.append(2*p + singles_spin)

        occ_idx = np.array(occ_idx, dtype=int)

        # Put all remaining spin–orbitals after the occupied block (preserving order).
        all_idx = np.arange(self.norb)
        virt_idx = all_idx[~np.isin(all_idx, occ_idx, assume_unique=False)]
        perm = np.concatenate([occ_idx, virt_idx])

        return perm
