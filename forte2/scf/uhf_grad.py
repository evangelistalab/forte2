import numpy as np

from forte2._forte2 import ints
from forte2.gradients import (
    build_metric_inverted_three_center,
    compute_gradient,
)
from forte2.system import ModelSystem


class UHFGradientMixin:
    """Density-fitted UHF analytic-gradient implementation."""

    def _build_df_deriv_weights(self, system, Cocc_a, Cocc_b, P):
        r"""
        Build density-fitted two-electron derivative weights for UHF.

        The returned weights contract directly with derivatives of ``(P|mn)`` and
        ``(P|Q)``. The metric inverse is applied as ``Z[P,m,n] = M^{-1}_{PQ}(Q|mn)``.

        Parameters
        ----------
        system : forte2.System
            The system for which to build the weights.
        Cocc_a : ndarray
            Occupied MO coefficients for alpha electrons with shape
            ``(nbasis, nocc_a)``.
        Cocc_b : ndarray
            Occupied MO coefficients for beta electrons with shape
            ``(nbasis, nocc_b)``.
        P : NDArray
            The AO density matrix with shape ``(nbasis, nbasis)``.

        Notes
        -----
        This code assumes we can store in memory the three-center integrals
        and the derivative weights. The memory requirement is thus
        ``O(naux * nbasis^2)`` and so the algorithm should be applicable to
        systems with 500-750 basis functions and 1000-1500 auxiliary
        functions. To scale this up, a direct approach is needed.
        """
        # compute Z[P,m,n] = M^{-1}_{PQ}(Q|mn)
        Z = build_metric_inverted_three_center(system)

        # compute rho[P] = P_mn Z[P,m,n]
        rho = np.einsum("mn,Pmn->P", P, Z, optimize=True)
        # compute Z_oo[P,i,j] = C_mi Z[P,m,n] C_nj
        Z_oo_a = np.einsum("mi,Pmn,nj->Pij", Cocc_a, Z, Cocc_a, optimize=True)
        Z_oo_b = np.einsum("mi,Pmn,nj->Pij", Cocc_b, Z, Cocc_b, optimize=True)

        # compute the two-electron derivative weights for the metric and three-center derivatives
        W2 = -0.5 * np.einsum("P,Q->PQ", rho, rho, optimize=True)
        W2 += 0.5 * np.einsum("Pij,Qji->PQ", Z_oo_a, Z_oo_a, optimize=True)
        W2 += 0.5 * np.einsum("Pij,Qji->PQ", Z_oo_b, Z_oo_b, optimize=True)

        W3 = np.einsum("mn,P->Pmn", P, rho, optimize=True)
        W3 -= np.einsum("mi,nj,Pji->Pmn", Cocc_a, Cocc_a, Z_oo_a, optimize=True)
        W3 -= np.einsum("mi,nj,Pji->Pmn", Cocc_b, Cocc_b, Z_oo_b, optimize=True)

        return W2, W3

    def gradient(self):
        """
        Compute the UHF analytic nuclear gradient with density fitting.

        Returns
        -------
        ndarray
            Gradient with shape ``(natoms, 3)`` in Hartree/Bohr.
        """
        self._validate_uhf_gradient_supported()

        if not self.executed:
            self.run()

        system = self.system
        Cocc_a = self.C[0][:, : self.na]
        Cocc_b = self.C[1][:, : self.nb]

        # Density matrix
        D1 = self.D[0] + self.D[1]
        # Energy-weighted density matrix
        # W1_mn = sum_i C_mi * eps_i * C_ni (i in occ_a) + sum_i C_mi * eps_i * C_ni (i in occ_b)
        W1 = np.einsum(
            "mi,i,ni->mn", Cocc_a, self.eps[0][: self.na], Cocc_a, optimize=True
        )
        W1 += np.einsum(
            "mi,i,ni->mn", Cocc_b, self.eps[1][: self.nb], Cocc_b, optimize=True
        )
        # Two-electron derivative weights for density fitting
        W2, W3 = self._build_df_deriv_weights(system, Cocc_a, Cocc_b, D1)

        # Gradient computation
        gradient = compute_gradient(system, D1, W1, W2, W3)

        return gradient

    def _validate_uhf_gradient_supported(self):
        """Reject UHF gradient cases outside the first DF implementation scope."""
        system = self.system

        if isinstance(system, ModelSystem):
            raise NotImplementedError(
                "UHF gradients are not implemented for ModelSystem."
            )
        if system.cholesky_tei:
            raise NotImplementedError(
                "UHF gradients are implemented only for density fitting, not cholesky_tei."
            )
        if system.use_gaussian_charges:
            raise NotImplementedError(
                "UHF gradients with Gaussian nuclear charges are not implemented."
            )
        if system.x2c_type is not None:
            raise NotImplementedError("UHF gradients with X2C are not implemented.")
        if system.auxiliary_basis is None:
            raise NotImplementedError(
                "UHF gradients require an auxiliary basis set for density fitting."
            )

        max_l = max(system.basis.max_l, system.auxiliary_basis.max_l)
        if max_l > ints.libint2_max_am:
            raise NotImplementedError(
                "UHF gradients require derivative integrals supported by Libint2 "
                f"(max_l = {max_l}, Libint2 max_l = {ints.libint2_max_am})."
            )
