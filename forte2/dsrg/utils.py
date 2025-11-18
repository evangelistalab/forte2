import numpy as np
from forte2 import dsrg_utils

compute_t1_block = dsrg_utils.compute_T1_block
compute_t2_block = dsrg_utils.compute_T2_block
renormalize_V_block = dsrg_utils.renormalize_V_block
renormalize_3index = dsrg_utils.renormalize_3index


def antisymmetrize_2body(T, indices):
    # antisymmetrize the residual
    T_anti = np.zeros(T.shape, dtype="complex128")
    T_anti += np.einsum("ijab->ijab", T)
    if indices[0] == indices[1]:
        T_anti -= np.einsum("ijab->jiab", T)
        if indices[2] == indices[3]:
            T_anti += np.einsum("ijab->jiba", T)
    if indices[2] == indices[3]:
        T_anti -= np.einsum("ijab->ijba", T)
    return T_anti


def cas_energy_given_cumulants(E_core, H_cas, V_cas, gamma1, gamma2):
    # see eq B.3. of JCP 146, 124132 (2017), but instead of gamma2, use lambda2
    e1 = np.einsum("uv,uv->", H_cas, gamma1, optimize=True)
    e2 = 0.5 * np.einsum("uvxy,uvxy->", V_cas, gamma2, optimize=True)
    return E_core + e1 + e2
