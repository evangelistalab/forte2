from dataclasses import dataclass

import numpy as np

from forte2.system.basis_utils import BasisInfo
from forte2.system import ModelSystem
from forte2.helpers import logger
from forte2.symmetry import real_sph_to_j_adapted
from forte2.helpers import canonical_orth
from .scf_base import SCFBase
from .rhf import RHF
from .scf_utils import guess_mix_ghf, alpha_beta_mix, break_complex_conjugation_symmetry


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
    alpha_beta_mix : bool, optional, default=False
        If True, will mix the highest two spinorbitals to try to seed alpha-beta orbital gradients.
    break_complex_symmetry : bool, optional, default=False
        If True, will add a small complex perturbation to the initial density matrix. This will both break
        the complex conjugation symmetry and Sz symmetry (allowing alpha-beta density matrix blocks to be nonzero)
    j_adapt: bool, optional, default=False
        If True, the j-adapted spinor AO basis will be used instead of the spherical AO basis.
    """

    ms_guess: float = None
    guess_mix: bool = False  # only used if nel is even
    alpha_beta_mix: bool = False
    break_complex_symmetry: bool = False
    j_adapt: bool = False

    _diis_update = RHF._diis_update
    _assign_orbital_symmetries = RHF._assign_orbital_symmetries

    def __call__(self, system):
        system.two_component = True
        if self.j_adapt:
            ua, ub = real_sph_to_j_adapted(system.basis)
            self.Usph2j = np.vstack((ua, ub))
            S = system.ints_overlap()
            S_spinor = self.Usph2j.conj().T @ S @ self.Usph2j
            self.Xorth_spinor, _, info = canonical_orth(S_spinor, system.ortho_thresh)
            self.nmo_spinor = info["n_kept"]
        self = super().__call__(system)
        self._parse_state()
        return self

    def _parse_state(self):
        if self.ms_guess is not None:
            assert np.isclose(
                int(round(self.ms_guess * 2)), self.ms_guess * 2
            ), "ms_guess must be a multiple of 0.5."
            self.twicems_guess = int(round(self.ms_guess * 2))
            if self.nel % 2 != self.twicems_guess % 2:
                raise ValueError(
                    f"{self.nel} electrons is incompatible with ms_guess={self.ms_guess}!"
                )
            self.na_guess = int(round(self.nel + self.twicems_guess) / 2)
            self.nb_guess = int(round(self.nel - self.twicems_guess) / 2)
            assert (
                self.nel == self.na_guess + self.nb_guess
            ), f"Number of electrons {self.nel} does not match na + nb = {self.na_guess} + {self.nb_guess}."
            assert (
                self.na_guess >= 0 and self.nb_guess >= 0
            ), f"{self._scf_type} requires non-negative number of alpha and beta electrons."

    def _build_fock(self, H, fock_builder, S):
        Jaa, Jbb = fock_builder.build_J([self.D[0], self.D[3]])
        nbf = Jaa.shape[0]
        if self.iter == 0 and self.ms_guess is not None:
            # Apply na/nb_guess
            mo_a, mo_b = self._guess_ms(self.C[0])
            occ = list(mo_a[: self.na_guess]) + list(mo_b[: self.nb_guess])
            occ = sorted(occ)
            Kaa, Kab, Kba, Kbb = fock_builder.build_K([self.C[0][:, occ]])
        else:
            Kaa, Kab, Kba, Kbb = fock_builder.build_K([self.C[0][:, : self.nel]])
        F = H.copy()
        F[:nbf, :nbf] += Jaa + Jbb - Kaa
        F[:nbf, nbf:] += -Kab
        F[nbf:, :nbf] += -Kba
        F[nbf:, nbf:] += Jaa + Jbb - Kbb

        F_canon = F

        return [F], [F_canon]

    def _build_density_matrix(self):
        # D = Cocc Cocc^+
        if self.iter == 0 and self.ms_guess is not None:
            # apply na/nb_guess
            occ_a, occ_b = self._guess_ms(self.C[0])
            Ca = self.C[0][:, occ_a[: self.na_guess]]
            Cb = self.C[0][:, occ_b[: self.nb_guess]]
            D = np.einsum("mi,ni->mn", Ca, Ca.conj())
            D += np.einsum("mi,ni->mn", Cb, Cb.conj())
        else:
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

        return Daa, Dab, Dba, Dbb

    def _build_total_density_matrix(self):
        Daa, Dab, Dba, Dbb = self._build_density_matrix()
        return np.block([[Daa, Dab], [Dba, Dbb]])

    def _apply_level_shift(self, F, S):
        if self.level_shift is None or self.level_shift < 1e-4:
            return F
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        D_vir = S - S @ D_spinor @ S

        return [F[0] + self.level_shift * D_vir]

    def _initial_guess(self, H, guess_type="minao"):
        C = RHF._initial_guess(self, H, guess_type)[0]
        if self.guess_mix and self.ms_guess is not None:
            if self.twicems_guess == 0:
                mo_a, mo_b = self._guess_ms(C)
                C = guess_mix_ghf(
                    C,
                    mo_a[self.na_guess - 1],
                    mo_b[self.nb_guess - 1],
                    mo_a[self.na_guess],
                    mo_b[self.nb_guess],
                )
        if self.alpha_beta_mix:
            C = alpha_beta_mix(C)
        if self.break_complex_symmetry:
            C = break_complex_conjugation_symmetry(C)
        return [C]

    def _build_ao_grad(self, S, F):
        Daa, Dab, Dba, Dbb = self.D
        D_spinor = np.block([[Daa, Dab], [Dba, Dbb]])
        sdf = S @ D_spinor @ F[0]
        AO_grad = sdf.conj().T - sdf
        AO_grad = self.Xorth.conj().T @ AO_grad @ self.Xorth

        return AO_grad

    def _eigh(self, F):
        Xorth = self.Xorth_spinor if self.j_adapt else self.Xorth
        Ftilde = Xorth.conj().T @ F @ Xorth
        e, c = np.linalg.eigh(Ftilde)
        return e, Xorth @ c

    def _diagonalize_fock(self, F):
        if self.j_adapt:
            F_spinor = self.Usph2j.conj().T @ F[0] @ self.Usph2j
            eps, C = self._eigh(F_spinor)
            C = self.Usph2j @ C
        else:
            eps, C = self._eigh(F[0])
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
            string += f"{i:<4d} ({self.irrep_labels[i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = nocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += f"{idx:<4d} ({self.irrep_labels[idx]}) {self.eps[0][idx]:<12.6f} "
        logger.log_info1(string)

    def _guess_ms(self, C):
        nmo = C.shape[1]
        nbf = C.shape[0] // 2
        mo_a = []
        mo_b = []
        for i in range(nmo):
            norm_a = np.linalg.norm(C[:nbf, i])
            norm_b = np.linalg.norm(C[nbf:, i])
            if norm_a >= norm_b:
                mo_a.append(i)
            else:
                mo_b.append(i)
        return np.array(mo_a), np.array(mo_b)

    def _print_ao_composition(self):
        if isinstance(self.system, ModelSystem):
            # send a PR if you want this changed
            return
        basis_info = BasisInfo(self.system, self.system.basis)
        if self.system.x2c_type == "so":
            logger.log_info1("\nSpinor Composition of MOs (HOMO-6 to HOMO):")
            if not hasattr(self, "Usph2j"):
                ua, ub = real_sph_to_j_adapted(self.system.basis)
                self.Usph2j = np.vstack((ua, ub))
            C = self.Usph2j.conj().T @ self.C[0]
            basis_info.print_spinor_composition(
                C, list(range(max(self.nel - 6, 0), self.nel))
            )
            logger.log_info1("\nSpinor Composition of MOs (LUMO to LUMO+6):")
            basis_info.print_spinor_composition(
                C, list(range(self.nel, min(self.nel + 6, self.nmo * 2)))
            )
        else:
            logger.log_info1("\nAO Composition of MOs (HOMO-5 to HOMO):")
            basis_info.print_ao_composition(
                self.C[0], list(range(max(self.nel - 6, 0), self.nel)), spinorbital=True
            )
            logger.log_info1("\nAO Composition of MOs (LUMO to LUMO+5):")
            basis_info.print_ao_composition(
                self.C[0],
                list(range(self.nel, min(self.nel + 6, self.nmo * 2))),
                spinorbital=True,
            )
