import numpy as np

from forte2.state import StateAverageInfo
from forte2.helpers import logger


def spin_matrices(system, C, skip_picture_change=None):
    r"""
    Build the spin operator matrices in the spinor basis spanned by `C`.

    The matrices are assembled in the two-component AO basis and then transformed to the
    MO basis. That order lets the one-electron part of :math:`\hat{S}^2` resolve the
    identity over the complete AO space instead of over the spinors carried in `C`.

    Under X2C the returned :math:`\hat{s}_z`, :math:`\hat{s}_+`, and :math:`\hat{s}_-`
    carry the picture change correction, but `S2_1e` does not. `S2_1e` is the
    same-electron part of :math:`\hat{S}^2`, where :math:`\hat{s}^2 = 3/4` is an even
    operator, so its correction is 3/4 times that of the identity, which the
    renormalization matrix makes equal to the overlap. The untransformed matrices already
    reproduce that exactly. Building `S2_1e` from picture-changed factors would instead
    spoil it, because the upper-left block of a product of transformed operators is not
    the transform of their product.

    Parameters
    ----------
    system : System
        The two-component system, which supplies the AO overlap.
    C : NDArray
        The spinor coefficients, shape (2*nbf, nspinor).
    skip_picture_change : bool | None, optional, default=None
        If True, skip the picture change correction of the spin operator, only relevant
        for X2C calculations. If None, follow ``skip_picture_change`` on the system's
        X2C parameters.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        The matrices `S_z`, `S_plus`, `S_minus`, and `S2_1e`, each of shape
        (nspinor, nspinor). The first three represent :math:`\hat{s}_z`,
        :math:`\hat{s}_+`, and :math:`\hat{s}_-`; the last is the one-electron part of
        :math:`\hat{S}^2`.
    """
    nbf = system.nbf
    assert (
        C.shape[0] == 2 * nbf
    ), "C must be in the spinorbital basis, with shape (2*nbf, nspinor)."
    ovlp = system.ints_overlap()[:nbf, :nbf]

    # s_z is diagonal in the spin blocks, s_+ maps beta onto alpha
    S_z = np.zeros((2 * nbf, 2 * nbf), dtype=complex)
    S_z[:nbf, :nbf] = 0.5 * ovlp
    S_z[nbf:, nbf:] = -0.5 * ovlp
    S_plus = np.zeros((2 * nbf, 2 * nbf), dtype=complex)
    S_plus[:nbf, nbf:] = ovlp
    S_minus = S_plus.conj().T

    # S^2 = S_- S_+ + S_z^2 + S_z
    # For two one-body operators, the product can be separated into
    # one-body and a two-body parts:
    # A x B = (a_pq a+_p a_q) (b_rs a+_r a_s)
    #     = a_pq b_rs (delta_pr a+_p a_s - a+_p a+_r a_q a_s)
    #     = ((a@b)_ps) a+_p a_s - (a_pq b_rs) a+_p a+_r a_q a_s

    # X @ X^+ is S^{-1} with linear dependencies removed
    X = system.get_Xorth()
    ovlp_inv = X @ X.conj().T
    S2_1e = S_z @ ovlp_inv @ S_z + S_z + S_minus @ ovlp_inv @ S_plus

    if skip_picture_change is None:
        skip_picture_change = system.skip_picture_change

    if system.x2c_type in ["sf", "so"] and not skip_picture_change:
        s_x, s_y, s_z = system.x2c_helper.spin_operator()
        S_z = s_z
        S_plus = s_x + 1j * s_y
        S_minus = S_plus.conj().T

    return tuple(C.conj().T @ A @ C for A in (S_z, S_plus, S_minus, S2_1e))


def _split_blocks(A, ncore):
    """Split a spinor matrix into its core-core, core-active, active-core, and active-active blocks."""
    co, ac = slice(0, ncore), slice(ncore, None)
    return A[co, co], A[co, ac], A[ac, co], A[ac, ac]


def compute_spin2(system, C, g1, g2, skip_picture_change=None):
    r"""
    Compute <S^2>, <S_x/y/z> of a two-component CI state

    Parameters
    ----------
    system : System
        The two-component system, which supplies the AO overlap.
    C : NDArray
        The coefficients of the core and active spinors, shape (2*nbf, ncore + nactv), with
        the core columns first. The number of core spinors follows from the size of `g1`.
    g1 : NDArray
        The complex active-space one-particle RDM,
        :math:`\gamma_{pq} = \langle a^\dagger_p a_q \rangle`.
    g2 : NDArray
        The complex active-space two-particle RDM,
        :math:`\gamma_{pqrs} = \langle a^\dagger_p a^\dagger_q a_s a_r \rangle`.
    skip_picture_change : bool | None, optional, default=None
        If True, skip the picture change correction of the spin operator, only relevant
        for X2C calculations. If None, follow ``skip_picture_change`` on the system's
        X2C parameters.

    Returns
    -------
    spin2 : float
        The expectation value of :math:`\hat{S}^2`.
    spin_vector : NDArray
        The expectation values of :math:`\hat{S}_x`, :math:`\hat{S}_y`, and
        :math:`\hat{S}_z`.
    """
    ncore = C.shape[1] - g1.shape[0]
    S_z, S_plus, S_minus, S2_1e = spin_matrices(system, C, skip_picture_change)
    S2_1e_cc, _, _, S2_1e_aa = _split_blocks(S2_1e, ncore)

    # <S^2>_1e = [S2_1e]_pq g1[p,q] = Tr(S2_1e,core) + sum_uv [S2_1e,act]_uv g1[u,v]
    # (g1 has no core-active block: one one-body operator cannot move an electron out of
    # a spinor that every determinant occupies and land back on the same state)
    spin2 = np.trace(S2_1e_cc) + np.einsum("uv,uv->", S2_1e_aa, g1)

    # <S^2>_2e = - [(Sz)_ps (Sz)_qr + (S_-)_ps (S_+)_qr] g2[p,q,r,s], with
    # g2[p,q,r,s] = <p^+ q^+ s r>: the first matrix carries the outer index pair, the
    # second the inner one, so that each keeps its own (creation, annihilation) pair
    for A, B in ((S_z, S_z), (S_minus, S_plus)):
        Acc, Aca, Aac, Aaa = _split_blocks(A, ncore)
        Bcc, Bca, Bac, Baa = _split_blocks(B, ncore)
        # core-core
        # g2[i,j,k,l] = <i^+ j^+ l k> = d_ik d_jl - d_il d_jk
        two_body = np.trace(Acc @ Bcc) - np.trace(Acc) * np.trace(Bcc)
        # core-active
        # g2[i,u,j,v] = <i^+ u^+ v j> = +d_ij g1[u,v]
        # g2[u,i,v,j] = <u^+ i^+ j v> = +d_ij g1[u,v]
        # g2[i,u,w,j] = <i^+ u^+ j w> = -d_ij g1[u,w]
        # g2[u,i,j,v] = <u^+ i^+ v j> = -d_ij g1[u,v]
        heff = Bac @ Aca + Aac @ Bca - np.trace(Acc) * Baa - np.trace(Bcc) * Aaa
        two_body += np.einsum("uv,uv->", heff, g1)
        # active-active
        two_body += np.einsum("ux,vw,uvwx->", Aaa, Baa, g2, optimize=True)
        spin2 -= two_body

    # the spin vector is one-body, so it needs the one-particle RDM alone
    spin_vector = []
    for k in (0.5 * (S_plus + S_minus), -0.5j * (S_plus - S_minus), S_z):
        k_cc, _, _, k_aa = _split_blocks(k, ncore)
        spin_vector.append((np.trace(k_cc) + np.einsum("uv,uv->", k_aa, g1)).real)

    return spin2.real, np.array(spin_vector)


def pretty_print_rel_spin_summary(
    sa_info: StateAverageInfo,
    spin2: np.ndarray,
    spin_vector: np.ndarray,
    header="\nTwo-component spin summary",
):
    r"""
    Print the spin expectation values of the two-component CI roots.

    Parameters
    ----------
    sa_info : StateAverageInfo
        An instance of `StateAverageInfo` that holds information about the states and their
        roots.
    spin2 : NDArray
        :math:`\langle \hat{S}^2 \rangle` for each root.
    spin_vector : NDArray
        :math:`\langle \hat{S}_x \rangle`, :math:`\langle \hat{S}_y \rangle`, and
        :math:`\langle \hat{S}_z \rangle` for each root, shape (nroots, 3).
    header : str, optional, default="Two-component spin summary"
        A header string to display at the top of the summary.
    """
    logger.log_info1(f"{header}:")
    width = 54
    logger.log_info1("=" * width)
    logger.log_info1(
        f"{'Root':>6} {'<S^2>':>12} {'<Sx>':>11} {'<Sy>':>11} {'<Sz>':>11}"
    )
    logger.log_info1("-" * width)
    for iroot in range(sa_info.nroots_sum):
        sx, sy, sz = spin_vector[iroot]
        logger.log_info1(
            f"{iroot:>6d} {spin2[iroot]:>12.6f} {sx:>11.6f} {sy:>11.6f} {sz:>11.6f}"
        )
    logger.log_info1("=" * width)
