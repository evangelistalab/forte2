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


def hermitize_and_antisymmetrize_two_body_dense(T):
    # antisymmetrize the residual
    T += np.einsum(
        "ijab->abij", T.conj()
    )  # This is the Hermitized version (i.e., [H,A]), which should then be antisymmetrized
    temp = T.copy()
    T -= np.einsum("ijab->jiab", temp)
    T += np.einsum("ijab->jiba", temp)
    T -= np.einsum("ijab->ijba", temp)


def hermitize_and_antisymmetrize_two_body(T):
    blks = set(T.keys())
    # Hermitize first
    for blk in T.keys():
        if blk not in blks:
            continue
        herm_blk = blk[2:] + blk[:2]
        if herm_blk in T.keys():
            temp = T[blk].copy()
            T[blk] += T[herm_blk].transpose(2, 3, 0, 1).conj()
            T[herm_blk] += temp.transpose(2, 3, 0, 1).conj()
            blks.remove(blk)
            blks.remove(herm_blk)

    for blk in T.keys():
        ij_same = blk[0] == blk[1]
        kl_same = blk[2] == blk[3]
        if not (ij_same or kl_same):
            continue
        temp = T[blk].copy()
        if ij_same:
            T[blk] -= temp.transpose(1, 0, 2, 3)
        if kl_same:
            T[blk] -= temp.transpose(0, 1, 3, 2)
        if ij_same and kl_same:
            T[blk] += temp.transpose(1, 0, 3, 2)


def hermitize_one_body(T):
    blks = set(T.keys())
    for blk in T.keys():
        if blk not in blks:
            continue
        herm_blk = blk[1] + blk[0]
        if herm_blk in T.keys():
            temp = T[blk].T.conj()
            T[blk] += T[herm_blk].T.conj()
            T[herm_blk] += temp
            blks.remove(blk)
            blks.remove(herm_blk)


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
