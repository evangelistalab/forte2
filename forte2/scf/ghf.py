from dataclasses import dataclass
import numpy as np

from forte2.helpers import logger
from forte2.symmetry import assign_mo_symmetries
from .scf_base import SCFBase
from .scf_utils import guess_mix
from .rhf import RHF


@dataclass
class GHF(SCFBase):
    r"""
    Generalized Hartree-Fock (GHF) method.
    The GHF spinor basis is a direct product of the atomic basis and :math:`\{|\alpha\rangle, |\beta\rangle\}`:
    
    .. math::
    
        |\psi_i\rangle = \sum_{\mu} \sum_{\sigma\in\{\alpha,\beta\}} c^{\sigma}_{\mu i} |\chi_{\mu}\rangle\otimes|\sigma\rangle

    The MO coefficients are stored in a square array

    .. math::
        \mathbf{C} = \begin{bmatrix}
        \mathbf{c}^{\alpha}_0 & \mathbf{c}^{\alpha}_1 & \dots\\
        \mathbf{c}^{\beta}_0 & \mathbf{c}^{\beta}_1 & \dots
        \end{bmatrix}

    Parameters
    ----------
    guess_mix : bool, optional, default=False
        If True, will mix the HOMO and LUMO orbitals to try to break alpha-beta degeneracy if nel is even.
    break_complex_symmetry : bool, optional, default=False
        If True, will add a small complex perturbation to the initial density matrix. This will both break
        the complex conjugation symmetry and Sz symmetry (allowing alpha-beta density matrix blocks to be nonzero)
    """

    guess_mix: bool = False  # only used if nel is even
    break_complex_symmetry: bool = False

    _diis_update = RHF._diis_update

    def __call__(self, system):
        system.two_component = True
        self = super().__call__(system)
        return self

    def _build_fock(self, H, fock_builder, S):
        Jaa, Jbb = fock_builder.build_J([self.D[0], self.D[3]])
        nbf = Jaa.shape[0]
        if self.iter == 0 and self.break_complex_symmetry:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K_density(self.D)
        else:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K(
                [self.C[0][:nbf, : self.nel], self.C[0][nbf:, : self.nel]], cross=True
            )
        F = H.copy()
        F[:nbf, :nbf] += Jaa + Jbb - Kaa
        F[:nbf, nbf:] += -Kab
        F[nbf:, :nbf] += -Kba
        F[nbf:, nbf:] += Jaa + Jbb - Kbb

        F_canon = F

        return [F], [F_canon]

    def _build_density_matrix(self):
        # D = Cocc Cocc^+
        D = np.einsum(
            "mi,ni->mn",
            self.C[0][:, : self.nel],
            self.C[0][:, : self.nel].conj(),
        )
        nbf = self.nbf
        Daa = D[:nbf, :nbf]
        Dab = D[:nbf, nbf:]
        Dba = D[nbf:, :nbf]
        Dbb = D[nbf:, nbf:]

        if self.iter == 0 and self.break_complex_symmetry:
            Daa[0, :] += 0.1j
            Dab[0, :] += 0.1j
            Daa[:, 0] -= 0.1j
            Dba[:, 0] -= 0.1j
            Dbb[0, :] += 0.1j
            Dbb[:, 0] -= 0.1j

        return Daa, Dab, Dba, Dbb

    def _build_total_density_matrix(self):
        Daa, *_, Dbb = self._build_density_matrix()
        return Daa + Dbb

    def _initial_guess(self, H, guess_type="minao"):
        C = RHF._initial_guess(self, H, guess_type)[0]
        if self.nel % 2 == 0 and self.guess_mix:
            C = guess_mix(C, self.nel, twocomp=True)
        return [C]

    def _build_ao_grad(self, S, F):
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        sdf = S @ D_spinor @ F[0]
        AO_grad = sdf.conj().T - sdf

        return AO_grad

    def _diagonalize_fock(self, F):
        eps, C = self._eigh_spinor(F[0])
        return [eps], [C]

    def _spin(self, S):
        """
        S^2 = 0.5 * (S+S- + S-S+) + Sz^2, S+ = sum_i si+, S- = sum_i si-
        We make use of the Slater-Condon rules to compute <GHF|S^2|GHF>
        """
        S_1c = S[: self.nbf, : self.nbf]
        mo_a = self.C[0][: self.nbf, : self.nel]
        mo_b = self.C[0][self.nbf :, : self.nel]

        # MO basis overlap matrices
        saa = mo_a.conj().T @ S_1c @ mo_a
        sbb = mo_b.conj().T @ S_1c @ mo_b
        sab = mo_a.conj().T @ S_1c @ mo_b
        sba = sab.conj().T

        na = saa.trace()
        nb = sbb.trace()

        S2_diag = (na + nb) * 0.5
        S2_offdiag = sba.trace() * sab.trace() - np.einsum("ij,ji->", sba, sab)
        Sz2_diag = (na + nb) * 0.25
        Sz2_offdiag = 0.25 * ((na - nb) ** 2 - np.linalg.norm(saa - sbb) ** 2)
        return (S2_diag + S2_offdiag + Sz2_diag + Sz2_offdiag).real

    def _energy(self, H, F):
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        energy = 0.5 * np.einsum("vu,uv->", D_spinor, H) + 0.5 * np.einsum(
            "vu,uv->", D_spinor, F[0]
        )
        return energy.real

    def _get_occupation(self):
        self.nocc = self.nel
        self.nuocc = self.nmo * 2 - self.nocc

    def _print_orbital_energies(self):
        nocc = self.nel
        nuocc = self.nmo * 2 - nocc
        orb_per_row = 5
        logger.log_info1("---------------------")
        logger.log_info1("Spinor Energies [Eh]")
        logger.log_info1("---------------------")
        logger.log_info1("Occupied:")
        string = ""
        for i in range(nocc):
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{i+1:<4d} ({self.irrep_labels[i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = nocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx+1:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
            )
        logger.log_info1(string)

    def _assign_orbital_symmetries(self):
        S = self._get_overlap()
        self.irrep_labels, self.irrep_indices = assign_mo_symmetries(
            self.system, S, self.C[0]
        )
