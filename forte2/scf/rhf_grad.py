import numpy as np

from forte2._forte2 import ints
from forte2.gradients import (
    build_metric_inverted_three_center,
    compute_gradient,
)
from forte2.system import ModelSystem


class RHFGradientMixin:
    """Density-fitted RHF analytic-gradient implementation."""

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
