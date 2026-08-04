import numpy as np
from numpy.typing import NDArray

from forte2.gradients import compute_gradient

from .hf_grad import (
    _build_hf_df_deriv_weights,
    _validate_hf_gradient_supported,
)


def _compute_uhf_gradient(uhf) -> NDArray:
    """Compute the density-fitted UHF analytic nuclear gradient."""
    _validate_hf_gradient_supported(uhf.system, "UHF")

    if not uhf.executed:
        uhf.run()

    Cocc_a = uhf.C[0][:, : uhf.na]
    Cocc_b = uhf.C[1][:, : uhf.nb]
    D1 = uhf.D[0] + uhf.D[1]
    W1 = np.einsum("mi,i,ni->mn", Cocc_a, uhf.eps[0][: uhf.na], Cocc_a, optimize=True)
    W1 += np.einsum("mi,i,ni->mn", Cocc_b, uhf.eps[1][: uhf.nb], Cocc_b, optimize=True)
    W2, W3 = _build_hf_df_deriv_weights(uhf.system, (Cocc_a, Cocc_b), D1)

    hcore_deriv = (
        uhf.system.x2c_helper.hcore_deriv() if uhf.system.x2c_type is not None else None
    )
    return compute_gradient(uhf.system, D1, W1, W2, W3, hcore_deriv=hcore_deriv)
