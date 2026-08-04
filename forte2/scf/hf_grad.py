from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from forte2.lib import ints
from forte2.gradients import build_metric_inverted_three_center
from forte2.system import ModelSystem


def _build_hf_df_deriv_weights(
    system,
    occupied_coefficients: Iterable[NDArray],
    D1: NDArray,
) -> tuple[NDArray, NDArray]:
    r"""
    Build spin-resolved HF density-fitted two-electron derivative weights.

    Let :math:`\sigma` label the occupied spin blocks supplied in
    ``occupied_coefficients`` and define

    .. math::
        Z^{P,\sigma}_{ij}
        =
        C^\sigma_{\mu i} Z^P_{\mu\nu} C^\sigma_{\nu j},
        \qquad
        \rho^P = D_{\mu\nu} Z^P_{\mu\nu}.

    The metric and three-center derivative weights are

    .. math::
        W_{PQ}
        =
        -\frac{1}{2}\rho^P\rho^Q
        +
        \frac{1}{2}\sum_\sigma
        Z^{P,\sigma}_{ij}Z^{Q,\sigma}_{ji},

    and

    .. math::
        W^P_{\mu\nu}
        =
        D_{\mu\nu}\rho^P
        -
        \sum_\sigma
        C^\sigma_{\mu i}C^\sigma_{\nu j}Z^{P,\sigma}_{ji}.

    RHF is obtained by supplying its occupied coefficient matrix twice, once
    for each spin. UHF supplies distinct alpha and beta occupied blocks.

    Parameters
    ----------
    system : System
        Molecular system providing the AO and auxiliary bases.
    occupied_coefficients : Iterable[NDArray]
        Occupied MO coefficient matrices, one for each spin block.
    D1 : NDArray
        Total spin-summed AO one-particle density.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(W2, W3)`` with shapes ``(naux, naux)`` and
        ``(naux, nbasis, nbasis)``.
    """
    Z = build_metric_inverted_three_center(system)
    rho = np.einsum("mn,Pmn->P", D1, Z, optimize=True)

    W2 = -0.5 * np.einsum("P,Q->PQ", rho, rho, optimize=True)
    W3 = np.einsum("mn,P->Pmn", D1, rho, optimize=True)

    for Cocc in occupied_coefficients:
        Z_oo = np.einsum("mi,Pmn,nj->Pij", Cocc, Z, Cocc, optimize=True)
        W2 += 0.5 * np.einsum("Pij,Qji->PQ", Z_oo, Z_oo, optimize=True)
        W3 -= np.einsum("mi,nj,Pji->Pmn", Cocc, Cocc, Z_oo, optimize=True)

    return W2, W3


def _build_ghf_df_deriv_weights(
    system,
    occupied_spinors: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Build the spatial density and DF derivative weights for complex GHF.

    The occupied spinors are stored as stacked alpha and beta spatial blocks.
    The Coulomb term depends on their spin trace, while exchange depends on the
    full spin-summed occupied transition density

    .. math::
        Q^P_{ij} = \sum_{\omega \in \{\alpha,\beta\}}
        C^{\omega *}_{\mu i} Z^P_{\mu\nu} C^\omega_{\nu j}.

    Parameters
    ----------
    system : System
        Molecular system providing the orbital and auxiliary bases.
    occupied_spinors : NDArray
        Occupied GHF coefficients with shape ``(2 * nbasis, nocc)``.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The spin-traced spatial density ``D1`` and the metric and
        three-center derivative weights ``W2`` and ``W3``.
    """
    nbf = system.nbf
    if occupied_spinors.ndim != 2 or occupied_spinors.shape[0] != 2 * nbf:
        raise ValueError("occupied_spinors must have shape (2 * system.nbf, nocc).")

    Cocc_a = occupied_spinors[:nbf]
    Cocc_b = occupied_spinors[nbf:]
    D1 = Cocc_a @ Cocc_a.conj().T + Cocc_b @ Cocc_b.conj().T

    Z = build_metric_inverted_three_center(system)
    rho = np.einsum("mn,Pmn->P", D1, Z, optimize=True)

    Q = np.einsum("mi,Pmn,nj->Pij", Cocc_a.conj(), Z, Cocc_a, optimize=True)
    Q += np.einsum("mi,Pmn,nj->Pij", Cocc_b.conj(), Z, Cocc_b, optimize=True)

    W2 = -0.5 * np.einsum("P,Q->PQ", rho, rho, optimize=True)
    W2 += 0.5 * np.einsum("Pij,Qji->PQ", Q, Q, optimize=True)

    W3 = np.einsum("nm,P->Pmn", D1, rho, optimize=True)
    W3 -= np.einsum("mi,nj,Pji->Pmn", Cocc_a.conj(), Cocc_a, Q, optimize=True)
    W3 -= np.einsum("mi,nj,Pji->Pmn", Cocc_b.conj(), Cocc_b, Q, optimize=True)

    return D1, W2, W3


def _validate_hf_gradient_supported(system, method_name: str) -> None:
    """Validate system-level assumptions shared by SCF gradients."""
    if isinstance(system, ModelSystem):
        raise NotImplementedError(
            f"{method_name} gradients are not implemented for ModelSystem."
        )
    if system.cholesky_tei:
        raise NotImplementedError(
            f"{method_name} gradients are implemented only for density fitting, "
            "not cholesky_tei."
        )
    if system.auxiliary_basis is None:
        raise NotImplementedError(
            f"{method_name} gradients require an auxiliary basis set for density fitting."
        )

    max_l = max(system.basis.max_l, system.auxiliary_basis.max_l)
    if max_l > ints.libint2_max_am:
        raise NotImplementedError(
            f"{method_name} gradients require derivative integrals supported by "
            f"Libint2 (max_l = {max_l}, Libint2 max_l = {ints.libint2_max_am})."
        )
