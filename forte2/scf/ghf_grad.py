import numpy as np
from numpy.typing import NDArray

from forte2.gradients import compute_gradient
from forte2.gradients.validation import validate_df_gradient_system

from .hf_grad import _build_ghf_df_deriv_weights


def _compute_ghf_gradient(ghf) -> NDArray:
    """Compute the density-fitted complex GHF analytic nuclear gradient."""
    validate_df_gradient_system(ghf.system, "GHF")

    if not ghf.executed:
        ghf.run()

    Cocc = ghf.C[0][:, : ghf.nel]
    D1, W2, W3 = _build_ghf_df_deriv_weights(ghf.system, Cocc)

    nbf = ghf.system.nbf
    W_spinor = np.einsum(
        "mi,i,ni->mn",
        Cocc,
        ghf.eps[0][: ghf.nel],
        Cocc.conj(),
        optimize=True,
    )
    W1 = W_spinor[:nbf, :nbf] + W_spinor[nbf:, nbf:]

    if ghf.system.x2c_type is None:
        # Scalar AO derivative integrals are real and spin diagonal. Their
        # contraction depends only on the real parts of the spin traces.
        return compute_gradient(ghf.system, D1.real, W1.real, W2, W3)

    if ghf.system.x2c_type == "sf":
        x2c_density = D1
    else:
        x2c_density = Cocc @ Cocc.conj().T
    hcore_gradient = ghf.system.x2c_helper.hcore_gradient(x2c_density)

    return compute_gradient(
        ghf.system,
        D1.real,
        W1.real,
        W2,
        W3,
        hcore_gradient=hcore_gradient,
    )
