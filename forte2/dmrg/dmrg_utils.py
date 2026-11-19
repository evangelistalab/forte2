import numpy as np


def block2_pdm_to_rdm(pdm):
    """
    Convert a block2 n-particle density matrix, or one of its spin blocks, to forte2's
    index order.

    block2 orders the annihilation indices in reverse, so the two layouts differ by
    reversing the annihilation axes.

    Parameters
    ----------
    pdm : NDArray
        The block2 PDM, ``pdm[p, q, ..., t, u] = <a^+_p a^+_q ... a_t a_u>``.

    Returns
    -------
    NDArray
        The RDM, ``gamma[p, q, ..., t, u] = <a^+_p a^+_q ... a_u a_t>``.
    """
    n = pdm.ndim // 2
    axes = list(range(n)) + list(range(pdm.ndim - 1, n - 1, -1))
    return np.ascontiguousarray(np.transpose(pdm, axes))
