from dataclasses import dataclass

import numpy as np

from forte2.system.basis_utils import BasisInfo
from forte2.system import ModelSystem
from forte2.helpers import logger
from .scf_base import SCFBase
from .rhf import RHF
from .uhf import UHF


@dataclass
class ROHF(SCFBase):
    """
    A class that runs restricted open-shell Hartree-Fock calculations.

    Parameters
    ----------
    ms : float
        Spin projection. Must be a multiple of 0.5.
    """

    ms: float = None

    _parse_state = UHF._parse_state
    _initial_guess = RHF._initial_guess
    _diagonalize_fock = RHF._diagonalize_fock
    _spin = RHF._spin
    _energy = UHF._energy
    _diis_update = RHF._diis_update
    _build_total_density_matrix = UHF._build_total_density_matrix
    _assign_orbital_symmetries = RHF._assign_orbital_symmetries
    _apply_level_shift = RHF._apply_level_shift

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _build_fock(self, H, fock_builder, S):
        Ja, Jb = fock_builder.build_J(self.D)
        K = fock_builder.build_K([self.C[0][:, : self.na], self.C[0][:, : self.nb]])
        F = [H + Ja + Jb - k for k in K]

        F_canon = self._build_canonical_fock(F, S)

        return F, F_canon

    def _build_canonical_fock(self, F, S):

        # Projection matrices for core (doubly occupied), open (singly occupied, alpha), and virtual (unoccupied) MO spaces
        U_core = np.dot(self.D[1], S)
        U_open = np.dot(self.D[0] - self.D[1], S)
        U_virt = np.eye(self.nbf) - np.dot(self.D[0], S)

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
        return [fock]

    def _build_density_matrix(self):
        D_a = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        D_b = np.einsum("mi,ni->mn", self.C[0][:, : self.nb], self.C[0][:, : self.nb])
        return D_a, D_b

    def _build_ao_grad(self, S, F):
        Deff = 0.5 * (self.D[0] + self.D[1])
        ao_grad = F @ Deff @ S - S @ Deff @ F
        ao_grad = self.Xorth.T @ ao_grad @ self.Xorth
        return ao_grad

    def _get_occupation(self):
        self.ndocc = min(self.na, self.nb)
        self.nsocc = abs(self.na - self.nb)
        self.nuocc = self.nmo - self.ndocc - self.nsocc

    def _print_orbital_energies(self):
        ndocc = min(self.na, self.nb)
        nsocc = abs(self.na - self.nb)
        nuocc = self.nmo - ndocc - nsocc
        orb_per_row = 5
        logger.log_info1("---------------------")
        logger.log_info1("Orbital Energies [Eh]")
        logger.log_info1("---------------------")
        logger.log_info1("Doubly Occupied:")
        string = ""
        for i in range(ndocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i:<4d} ({self.irrep_labels[i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        if nsocc > 0:
            logger.log_info1("\nSingly Occupied:")
            string = ""
            for i in range(nsocc):
                idx = ndocc + i
                if i % orb_per_row == 0:
                    string += "\n"
                string += (
                    f"{idx:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
                )
            logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = ndocc + nsocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
        logger.log_info1(string)

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        logger.log_info1("\nAO Composition of doubly occupied MOs (HOMO-3 to HOMO):")
        basis_info.print_ao_composition(
            self.C[0], list(range(max(self.ndocc - 3, 0), self.ndocc))
        )
        logger.log_info1("\nAO Composition of singly occupied MOs:")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.ndocc, self.ndocc + self.nsocc))
        )
        logger.log_info1("\nAO Composition of MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[0],
            list(
                range(
                    self.ndocc + self.nsocc, min(self.ndocc + self.nsocc + 5, self.nmo)
                )
            ),
        )
