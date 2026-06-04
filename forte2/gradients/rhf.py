import numpy as np
import scipy as sp

import forte2.integrals as integrals
from forte2._forte2 import ints
from forte2.helpers.matrix_functions import _eigh_metric_kernel
from forte2.system import ModelSystem, system

from .utils import flat_to_atom_gradient, nuclear_repulsion_deriv, compute_gradient

def rhf_gradient(rhf):
    """
    Compute the RHF analytic nuclear gradient with density fitting.

    Parameters
    ----------
    rhf : RHF
        Restricted Hartree-Fock object. If it has not been run, ``run()`` is called.

    Returns
    -------
    ndarray
        Gradient with shape ``(natoms, 3)`` in Hartree/Bohr.
    """
    _validate_rhf_gradient_supported(rhf)

    if not rhf.executed:
        rhf.run()

    system = rhf.system
    Cocc = rhf.C[0][:, : rhf.na]
    P = 2.0 * rhf.D[0]

    # Evaluate the energy-weighted density matrix
    # W_mn = 2 * sum_i C_mi * eps_i * C_ni (i in occ)
    W1 = 2.0 * np.einsum(
        "mi,i,ni->mn", Cocc, rhf.eps[0][: rhf.na], Cocc, optimize=True
    )
    # Build the two-electron derivative weights and contract with the integrals.
    W3, W2 = build_rhf_df_deriv_weights(system, Cocc, P)

    gradient = compute_gradient(system, P, W1, W2, W3)

    return gradient


def build_rhf_df_deriv_weights(system, Cocc, P):
    r"""
    Build density-fitted two-electron derivative weights for RHF.

    The returned weights contract directly with derivatives of ``(P|mn)`` and
    ``(P|Q)``. The metric inverse is applied as ``Z[P,m,n] = M^{-1}_{PQ}(Q|mn)``.

    Parameters
    ----------
    system : System
        The system for which to build the weights.
    Cocc : ndarray
        Occupied MO coefficients with shape ``(nbasis, nocc)``.
    P : ndarray
        The AO density matrix with shape ``(nbasis, nbasis)``.

    Notes
    -----
    This code assumes we can store in memory the three-center integrals
    and the derivative weights. The memory requirement is thus ``O(naux * nbasis^2)``
    and so the algorithm should be applicable to systems with 500-750 basis functions
    and 1000-1500 auxiliary functions. To scale this up, a direct approach is needed.
    """
    J = integrals.coulomb_3c(system, system.auxiliary_basis, system.basis, system.basis)
    M = integrals.coulomb_2c(system, system.auxiliary_basis, system.auxiliary_basis)
    Z = _apply_inverse_metric(system, M, J)

    rho = np.einsum("mn,Pmn->P", P, Z, optimize=True)
    Z_oo = np.einsum("mi,Pmn,nj->Pij", Cocc, Z, Cocc, optimize=True)

    W3 = np.einsum("mn,P->Pmn", P, rho, optimize=True)
    W3 -= 2.0 * np.einsum("mi,nj,Pji->Pmn", Cocc, Cocc, Z_oo, optimize=True)

    W2 = -0.5 * np.einsum("P,Q->PQ", rho, rho, optimize=True)
    W2 += np.einsum("Pij,Qji->PQ", Z_oo, Z_oo, optimize=True)

    return W3, W2


def _apply_inverse_metric(system, M, J):
    """Apply the density fitting metric inverse to a three-center tensor."""
    rhs = J.reshape(J.shape[0], -1)

    if system.df_ortho_rtol is None:
        try:
            L = sp.linalg.cholesky(M, lower=True)
        except sp.linalg.LinAlgError as exc:
            raise ValueError(
                "Density fitting Coulomb metric (P|Q) is not positive definite.\n"
                "Please set df_ortho_rtol to a small positive value to orthogonalize the metric."
            ) from exc
        result = sp.linalg.cho_solve((L, True), rhs)
    else:
        evals, evecs, info = _eigh_metric_kernel(M, rtol=system.df_ortho_rtol)
        ndiscard = info["n_discarded"]
        evals = evals[ndiscard:]
        evecs = evecs[:, ndiscard:]
        result = evecs @ ((evecs.T @ rhs) / evals[:, None])

    return result.reshape(J.shape)


def _validate_rhf_gradient_supported(rhf):
    """Reject RHF gradient cases outside the first DF implementation scope."""
    system = rhf.system

    if isinstance(system, ModelSystem):
        raise NotImplementedError("RHF gradients are not implemented for ModelSystem.")
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
