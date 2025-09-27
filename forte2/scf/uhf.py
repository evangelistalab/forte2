from dataclasses import dataclass

import numpy as np

from forte2.system.basis_utils import BasisInfo
from forte2.system import ModelSystem
from forte2.helpers import logger
# from forte2.symmetry import assign_mo_symmetries
from .scf_base import SCFBase
from .rhf import RHF
from .scf_utils import guess_mix


@dataclass
class UHF(SCFBase):
    """
    A class that runs unrestricted Hartree-Fock calculations.

    Parameters
    ----------
    ms : float
        Spin projection. Must be a multiple of 0.5.
    guess_mix : bool, optional, default=False
        If True, will mix the HOMO and LUMO orbitals to try to break alpha-beta degeneracy if ms is 0.0.
    """

    ms: float = None
    guess_mix: bool = False  # only used if ms == 0

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _parse_state(self):
        if self.ms is None:
            raise ValueError(
                f"Spin projection (ms) must be set for {self._scf_type()}."
            )
        assert np.isclose(
            int(round(self.ms * 2)), self.ms * 2
        ), "ms must be a multiple of 0.5."
        self.twicems = int(round(self.ms * 2))
        if self.nel % 2 != self.twicems % 2:
            raise ValueError(f"{self.nel} electrons is incompatible with ms={self.ms}!")
        self.na = int(round(self.nel + self.twicems) / 2)
        self.nb = int(round(self.nel - self.twicems) / 2)
        assert (
            self.nel == self.na + self.nb
        ), f"Number of electrons {self.nel} does not match na + nb = {self.na} + {self.nb}."
        assert (
            self.na >= 0 and self.nb >= 0
        ), f"{self._scf_type} requires non-negative number of alpha and beta electrons."

    def _build_fock(self, H, fock_builder, S):
        Ja, Jb = fock_builder.build_J(self.D)
        K = fock_builder.build_K([self.C[0][:, : self.na], self.C[1][:, : self.nb]])
        F = [H + Ja + Jb - k for k in K]

        F_canon = F

        return F, F_canon

    def _build_density_matrix(self):
        D_a = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        D_b = np.einsum("mi,ni->mn", self.C[1][:, : self.nb], self.C[1][:, : self.nb])
        return D_a, D_b

    def _build_total_density_matrix(self):
        D_a, D_b = self._build_density_matrix()
        return D_a + D_b

    def _initial_guess(self, H, guess_type="minao"):
        C = RHF._initial_guess(self, H, guess_type=guess_type)[0]

        if self.twicems == 0 and self.guess_mix:
            return guess_mix(C, self.nel // 2 - 1)

        return [C, C]

    def _build_ao_grad(self, S, F):
        AO_grad = np.hstack(
            [
                (self.Xorth.T @ (f @ d @ S - S @ d @ f) @ self.Xorth).flatten()
                for d, f in zip(self.D, F)
            ]
        )
        return AO_grad

    def _diagonalize_fock(self, F):
        eps_a, C_a = self._eigh(F[0])
        eps_b, C_b = self._eigh(F[1])
        return [eps_a, eps_b], [C_a, C_b]

    def _spin(self, S):
        # alpha-beta orbital overlap matrix
        # S_ij = < psi_i | psi_j >, i,j=occ
        #      = \sum_{uv} c_ui^* c_vj <u|v>
        S_ij = np.einsum(
            "ui,uv,vj->ij",
            self.C[0][:, : self.na].conj(),
            S,
            self.C[1][:, : self.nb],
            optimize=True,
        )
        # Spin contamination: <s^2> - <s^2>_exact = N_b - \sum_{ij} |S_ij|^2
        ds2 = self.nb - np.einsum("ij,ij->", S_ij.conj(), S_ij)
        # <S^2> value
        s2 = self.ms * (self.ms + 1) + ds2
        return s2

    def _energy(self, H, F):
        energy = 0.5 * (
            np.einsum("vu,uv->", self.D[0] + self.D[1], H)
            + np.einsum("vu,uv->", self.D[0], F[0])
            + np.einsum("vu,uv->", self.D[1], F[1])
        )
        return energy

    def _diis_update(self, diis, F, AO_grad):
        F_flat = diis.update(np.hstack([f.flatten() for f in F]), AO_grad)
        F = [
            F_flat[: self.nbf**2].reshape(self.nbf, self.nbf),
            F_flat[self.nbf**2 :].reshape(self.nbf, self.nbf),
        ]
        return F

    def _apply_level_shift(self, F, S):
        if self.level_shift is None or all(ls < 1e-4 for ls in self.level_shift):
            return F
        D_vir = [S - S @ d @ S for d in self.D]

        return [f + ls * d for ls, f, d in zip(self.level_shift, F, D_vir)]

    def _get_occupation(self):
        self.aocc = self.na
        self.auocc = self.nmo - self.aocc
        self.bocc = self.nb
        self.buocc = self.nmo - self.bocc

    def _print_orbital_energies(self):
        naocc = self.na
        naucc = self.nmo - naocc
        nbocc = self.nb
        nbucc = self.nmo - nbocc
        orb_per_row = 5
        logger.log_info1("---------------------")
        logger.log_info1("Orbital Energies [Eh]")
        logger.log_info1("---------------------")
        logger.log_info1("Alpha Occupied:")
        string = ""
        for i in range(naocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[0][i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nAlpha Virtual:")
        string = ""
        for i in range(naucc):
            idx = naocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[0][idx]}) {self.eps[0][idx]:<12.6f} "
            )
        logger.log_info1(string)

        logger.log_info1("\nBeta Occupied:")
        string = ""
        for i in range(nbocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[1][i]}) {self.eps[1][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nBeta Virtual:")
        string = ""
        for i in range(nbucc):
            idx = nbocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[1][idx]}) {self.eps[1][idx]:<12.6f} "
            )
        logger.log_info1(string)

    def _assign_orbital_symmetries(self):
        S = self._get_overlap()
        irrep_labels_alpha, irrep_indices_alpha = assign_mo_symmetries(
            self.system,
            self.basis_info,
            S,
            self.C[0],
        )
        irrep_labels_beta, irrep_indices_beta = assign_mo_symmetries(
            self.system,
            self.basis_info,
            S,
            self.C[1],
        )
        self.irrep_labels = [irrep_labels_alpha, irrep_labels_beta]
        self.irrep_indices = [irrep_indices_alpha, irrep_indices_beta]

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        logger.log_info1("\nAO Composition of Alpha MOs (HOMO-5 to HOMO):")
        basis_info.print_ao_composition(
            self.C[0], list(range(max(self.na - 5, 0), self.na))
        )
        logger.log_info1("\nAO Composition of Alpha MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.na, min(self.na + 5, self.nmo)))
        )
        logger.log_info1("\nAO Composition of Beta MOs (HOMO-5 to HOMO):")
        basis_info.print_ao_composition(
            self.C[1], list(range(max(self.na - 5, 0), self.na))
        )
        logger.log_info1("\nAO Composition of Beta MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[1], list(range(self.na, min(self.na + 5, self.nmo)))
        )
