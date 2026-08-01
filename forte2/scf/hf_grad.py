from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from forte2._forte2 import ints
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


def _validate_hf_gradient_supported(system, method_name: str) -> None:
    """Validate system-level assumptions shared by RHF and UHF gradients."""
    if isinstance(system, ModelSystem):
        raise NotImplementedError(
            f"{method_name} gradients are not implemented for ModelSystem."
        )
    if system.cholesky_tei:
        raise NotImplementedError(
            f"{method_name} gradients are implemented only for density fitting, "
            "not cholesky_tei."
        )
    if system.use_gaussian_charges:
        raise NotImplementedError(
            f"{method_name} gradients with Gaussian nuclear charges are not implemented."
        )
    if system.x2c_type is not None:
        raise NotImplementedError(
            f"{method_name} gradients with X2C are not implemented."
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
