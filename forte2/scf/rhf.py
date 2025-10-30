from dataclasses import dataclass
import numpy as np

from forte2.system.basis_utils import BasisInfo
from forte2.system import ModelSystem
from forte2.helpers import logger
from forte2.symmetry import MOSymmetryDetector
from .scf_base import SCFBase
from .scf_utils import minao_initial_guess, core_initial_guess


@dataclass
class RHF(SCFBase):
    """
    A class that runs restricted Hartree-Fock calculations.
    """

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _parse_state(self):
        assert self.nel % 2 == 0, "RHF requires an even number of electrons."
        self.ms = 0
        self.na = self.nb = self.nel // 2

    def _build_fock(self, H, fock_builder, S):
        J = fock_builder.build_J(self.D)[0]
        K = fock_builder.build_K([self.C[0][:, : self.na]])[0]
        F = H + 2.0 * J - K
        return [F], [F]

    def _build_density_matrix(self):
        D = np.einsum("mi,ni->mn", self.C[0][:, : self.na], self.C[0][:, : self.na])
        return [D]

    def _build_total_density_matrix(self):
        # returns the total density matrix (Daa + Dbb)
        return 2 * self._build_density_matrix()[0]

    def _initial_guess(self, H, guess_type="minao"):
        match guess_type:
            case "minao":
                C = minao_initial_guess(self.system, H)
            case "hcore":
                C = core_initial_guess(self.system, H)
            case _:
                raise RuntimeError(f"Unknown initial guess type: {guess_type}")

        return [C]

    def _build_ao_grad(self, S, F):
        ao_grad = F[0] @ self.D[0] @ S - S @ self.D[0] @ F[0]
        ao_grad = self.Xorth.T @ ao_grad @ self.Xorth
        return ao_grad

    def _energy(self, H, F):
        return np.sum(self.D[0] * (H + F[0]))

    def _diagonalize_fock(self, F):
        eps, C = self._eigh(F[0])
        return [eps], [C]

    def _spin(self, S):
        return self.ms * (self.ms + 1)

    def _diis_update(self, diis, F, AO_grad):
        return [diis.update(F[0], AO_grad)]

    def _apply_level_shift(self, F, S):
        if self.level_shift is None or self.level_shift < 1e-4:
            return F
        D_vir = S - S @ self.D[0] @ S

        return [F[0] + self.level_shift * D_vir]

    def _get_occupation(self):
        self.ndocc = self.na
        self.nuocc = self.nmo - self.ndocc

    def _print_orbital_energies(self):
        ndocc = self.na
        nuocc = self.nmo - ndocc
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

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = ndocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
        logger.log_info1(string)

    def _post_process(self):
        super()._post_process()
        self._print_ao_composition()

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        logger.log_info1("\nAO Composition of MOs (HOMO-5 to HOMO):")
        basis_info.print_ao_composition(
            self.C[0], list(range(max(self.na - 5, 0), self.na))
        )
        logger.log_info1("\nAO Composition of MOs (LUMO to LUMO+5):")
        basis_info.print_ao_composition(
            self.C[0], list(range(self.na, min(self.na + 5, self.nmo)))
        )

    def _assign_orbital_symmetries(self):
        S = self._get_overlap()
        mosym = MOSymmetryDetector(
            self.system,
            self.basis_info,
            S,
            self.C[0],
            self.eps[0],
        )
        mosym.run()
        self.irrep_labels = mosym.labels
        self.irrep_indices = mosym.irrep_indices
