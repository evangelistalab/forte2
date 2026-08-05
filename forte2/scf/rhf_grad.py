import numpy as np
from numpy.typing import NDArray

from forte2.gradients import compute_gradient

from .hf_grad import (
    _build_hf_df_deriv_weights,
    _validate_hf_gradient_supported,
)


def _compute_rhf_gradient(rhf) -> NDArray:
    """Compute the density-fitted RHF analytic nuclear gradient."""
    _validate_hf_gradient_supported(rhf.system, "RHF")

    if not rhf.executed:
        rhf.run()

    Cocc = rhf.C[0][:, : rhf.na]
    D1 = 2.0 * rhf.D[0]
    W1 = 2.0 * np.einsum("mi,i,ni->mn", Cocc, rhf.eps[0][: rhf.na], Cocc, optimize=True)
    W2, W3 = _build_hf_df_deriv_weights(rhf.system, (Cocc, Cocc), D1)

    hcore_gradient = (
        rhf.system.x2c_helper.hcore_gradient(D1)
        if rhf.system.x2c_type is not None
        else None
    )
    return compute_gradient(
        rhf.system,
        D1,
        W1,
        W2,
        W3,
        hcore_gradient=hcore_gradient,
    )
