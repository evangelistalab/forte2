import numpy as np
from forte2.lib import dsrg_utils

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


def degno_active_sf(e_dsrg, hbar1, hbar2, gamma1, lambda2):
    r"""
    De-normal-order a spin-free active-space effective Hamiltonian.

    Converts the normal-ordered :math:`\bar{H}` returned by the DSRG commutator
    machinery into the scalar/one-body/two-body triple a CI solver expects. The
    two-body part is unchanged; only the scalar and the one-body part absorb the
    contractions with the reference density.

    Parameters
    ----------
    e_dsrg : float
        The DSRG energy the scalar term is built on.
    hbar1 : np.ndarray
        The active-space one-body effective Hamiltonian, already Hermitized and
        including the bare Fock contribution.
    hbar2 : np.ndarray
        The active-space two-body effective Hamiltonian, already Hermitized and
        including the bare two-electron contribution.
    gamma1 : np.ndarray
        The 1-RDM of the reference in the active space.
    lambda2 : np.ndarray
        The 2-body cumulant of the reference in the active space.

    Returns
    -------
    tuple[float, np.ndarray]
        The scalar term and the de-normal-ordered one-body term. The inputs are
        left untouched, so this may be called more than once on the same tensors.
    """

    hbar2_temp = 2 * hbar2 - hbar2.swapaxes(2, 3)

    hbar0 = e_dsrg
    hbar0 -= np.einsum("vu,vu->", hbar1, gamma1, optimize=True)
    hbar0 += 0.25 * np.einsum("uv,vyux,xy->", gamma1, hbar2_temp, gamma1, optimize=True)
    hbar0 -= 0.5 * np.einsum("xyuv,uvxy->", hbar2, lambda2, optimize=True)

    hbar1_degno = hbar1 - 0.5 * np.einsum(
        "uxvy,yx->uv", hbar2_temp, gamma1, optimize=True
    )
    return hbar0, hbar1_degno


def rotate_active_ints(hbar1, hbar2, Uactv):
    """
    Rotate active-space integrals back into the basis the CI solver works in.

    Parameters
    ----------
    hbar1 : np.ndarray
        The one-body term in the semicanonical active basis.
    hbar2 : np.ndarray
        The two-body term in the semicanonical active basis.
    Uactv : np.ndarray
        The active-active block of the semicanonicalizing unitary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The one- and two-body terms in the original active basis.
    """

    hbar1_canon = np.einsum("ip,pq,jq->ij", Uactv, hbar1, Uactv.conj(), optimize=True)
    hbar2_canon = np.einsum(
        "ip,jq,pqrs,kr,ls->ijkl",
        Uactv,
        Uactv,
        hbar2,
        Uactv.conj(),
        Uactv.conj(),
        optimize=True,
    )
    return hbar1_canon, hbar2_canon
