from dataclasses import dataclass
import numpy as np

from forte2.system.basis_utils import BasisInfo
from forte2.system import ModelSystem
from forte2.helpers import logger
from forte2.symmetry import MOSymmetryDetector
from .scf_base import SCFBase
from .scf_utils import minao_initial_guess, core_initial_guess
from forte2._forte2 import ints
import forte2.integrals as integrals
from forte2.gradients import compute_gradient, build_metric_inverted_three_center


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

    def _build_df_deriv_weights(self, system, Cocc, P):
        r"""
        Build density-fitted two-electron derivative weights for RHF.

        The returned weights contract directly with derivatives of ``(P|mn)`` and
        ``(P|Q)``. The metric inverse is applied as ``Z[P,m,n] = M^{-1}_{PQ}(Q|mn)``.

        Parameters
        ----------
        system : forte2.System
            The system for which to build the weights.
        Cocc : ndarray
            Occupied MO coefficients with shape ``(nbasis, nocc)``.
        P : NDArray
            The AO density matrix with shape ``(nbasis, nbasis)``.

        Notes
        -----
        This code assumes we can store in memory the three-center integrals
        and the derivative weights. The memory requirement is thus ``O(naux * nbasis^2)``
        and so the algorithm should be applicable to systems with 500-750 basis functions
        and 1000-1500 auxiliary functions. To scale this up, a direct approach is needed.
        """
        # compute Z[P,m,n] = M^{-1}_{PQ}(Q|mn)
        Z = build_metric_inverted_three_center(system)

        # compute rho[P] = P_mn Z[P,m,n]
        rho = np.einsum("mn,Pmn->P", P, Z, optimize=True)
        # compute Z_oo[P,i,j] = C_mi Z[P,m,n] C_nj
        Z_oo = np.einsum("mi,Pmn,nj->Pij", Cocc, Z, Cocc, optimize=True)

        # compute the two-electron derivative weights for the metric and three-center derivatives
        W2 = -0.5 * np.einsum("P,Q->PQ", rho, rho, optimize=True)
        W2 += np.einsum("Pij,Qji->PQ", Z_oo, Z_oo, optimize=True)

        W3 = np.einsum("mn,P->Pmn", P, rho, optimize=True)
        W3 -= 2.0 * np.einsum("mi,nj,Pji->Pmn", Cocc, Cocc, Z_oo, optimize=True)

        return W2, W3

    def gradient(self):
        """
        Compute the RHF analytic nuclear gradient with density fitting.

        Returns
        -------
        ndarray
            Gradient with shape ``(natoms, 3)`` in Hartree/Bohr.
        """
        self._validate_rhf_gradient_supported()

        if not self.executed:
            self.run()

        system = self.system
        Cocc = self.C[0][:, : self.na]

        # Density matrix
        D1 = 2.0 * self.D[0]
        # Energy-weighted density matrix W1_mn = 2 * sum_i C_mi * eps_i * C_ni (i in occ)
        W1 = 2.0 * np.einsum(
            "mi,i,ni->mn", Cocc, self.eps[0][: self.na], Cocc, optimize=True
        )
        # Two-electron derivative weights for density fitting
        W2, W3 = self._build_df_deriv_weights(system, Cocc, D1)

        # Gradient computation
        gradient = compute_gradient(system, D1, W1, W2, W3)

        return gradient

    def _validate_rhf_gradient_supported(self):
        """Reject RHF gradient cases outside the first DF implementation scope."""
        system = self.system

        if isinstance(system, ModelSystem):
            raise NotImplementedError(
                "RHF gradients are not implemented for ModelSystem."
            )
        if system.cholesky_tei:
            raise NotImplementedError(
                "RHF gradients are implemented only for density fitting, not cholesky_tei."
            )
        if system.use_gaussian_charges:
            raise NotImplementedError(
                "RHF gradients with Gaussian nuclear charges are not implemented."
            )
        if system.x2c_type is not None:
            raise NotImplementedError("RHF gradients with X2C are not implemented.")
        if system.auxiliary_basis is None:
            raise NotImplementedError(
                "RHF gradients require an auxiliary basis set for density fitting."
            )

        max_l = max(system.basis.max_l, system.auxiliary_basis.max_l)
        if max_l > ints.libint2_max_am:
            raise NotImplementedError(
                "RHF gradients require derivative integrals supported by Libint2 "
                f"(max_l = {max_l}, Libint2 max_l = {ints.libint2_max_am})."
            )

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
            string += f"{i:<4d} ({self.irrep_labels[0][i]}) {self.eps[0][i]:<12.6f} "
        logger.log_info1(string)

        logger.log_info1("\nVirtual:")
        string = ""
        for i in range(nuocc):
            idx = ndocc + i
            if i % orb_per_row == 0:
                string += "\n"
            string += (
                f"{idx:<4d} ({self.irrep_labels[0][idx]}) {self.eps[0][idx]:<12.6f} "
            )
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
        self.irrep_labels = [mosym.labels]
        self.irrep_indices = [mosym.irrep_indices]
