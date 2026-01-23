import numpy as np
from forte2 import dsrg_utils

compute_t1_block = dsrg_utils.compute_T1_block
compute_t2_block = dsrg_utils.compute_T2_block
renormalize_V_block = dsrg_utils.renormalize_V_block
renormalize_3index = dsrg_utils.renormalize_3index


def antisymmetrize_2body(T, indices):
    # antisymmetrize the residual
    T_anti = np.zeros(T.shape, dtype=T.dtype)
    T_anti += np.einsum("ijab->ijab", T)
    if indices[0] == indices[1]:
        T_anti -= np.einsum("ijab->jiab", T)
        if indices[2] == indices[3]:
            T_anti += np.einsum("ijab->jiba", T)
    if indices[2] == indices[3]:
        T_anti -= np.einsum("ijab->ijba", T)
    return T_anti


def cas_energy_given_RDMs(E_core, H_cas, V_cas, gamma1, gamma2):
    r"""
    Return the CAS energy.

    .. math::
        E_{\mathrm{CAS}} = E_{\mathrm{core}} + \sum_{uv} \langle u | \hat{h} | v \rangle \gamma_v^u + \frac{1}{2} \sum_{uvxy} \langle uv | \hat{g} | xy \rangle \gamma_{xy}^{uv}

    Parameters
    ----------
    E_core : float
        The core energy.
    H_cas : np.ndarray
        The one-electron integrals in the CAS.
    V_cas : np.ndarray
        The two-electron integrals in the CAS (not antisymmetrized).
    gamma1 : np.ndarray
        The 1-RDM of the CAS reference.
    gamma2 : np.ndarray
        The 2-RDM of the CAS reference.

    Returns
    -------
    float
        The CAS energy.
    """

    e1 = np.einsum("uv,uv->", H_cas, gamma1, optimize=True)
    e2 = 0.5 * np.einsum("uvxy,uvxy->", V_cas, gamma2, optimize=True)
    return E_core + e1 + e2
