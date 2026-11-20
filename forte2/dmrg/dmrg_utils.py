import numpy as np


def block2_2pdm_to_sf_2rdm(pdm2):
    """
    Convert a block2 spin-adapted 2-particle density matrix to a forte2 spin-free 2-RDM.

    block2 orders the annihilation indices in reverse, so the two layouts differ by a
    swap of the last two axes.

    Parameters
    ----------
    pdm2 : NDArray
        The block2 2-PDM, ``pdm2[p, q, r, s] = <a^+_p a^+_q a_r a_s>`` summed over spin.

    Returns
    -------
    NDArray
        The spin-free 2-RDM, ``gamma2[p, q, r, s] = <a^+_p a^+_q a_s a_r>`` summed over
        spin.
    """
    return np.ascontiguousarray(np.transpose(pdm2, (0, 1, 3, 2)))
