import numpy as np

from forte2.state import StateAverageInfo
from forte2.helpers import logger


def spin_matrices(system, C, skip_picture_change=None):
    r"""
    Build the spin operator matrices in the spinor basis spanned by `C`.

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
    tuple[NDArray, NDArray, NDArray]
        The :math:`\hat{s}_z`, :math:`\hat{s}_+`, and :math:`\hat{s}_-` matrices, each of
        shape (nspinor, nspinor).
    """
    nbf = system.nbf
    assert (
        C.shape[0] == 2 * nbf
    ), "C must be in the spinorbital basis, with shape (2*nbf, nspinor)."

    if skip_picture_change is None:
        skip_picture_change = system.x2c.skip_picture_change

    if system.x2c.x2c_type in ["sf", "so"] and not skip_picture_change:
        s_x, s_y, S_z = system.x2c_helper.spin_operator()
        S_plus = s_x + 1j * s_y
    else:
        ovlp = system.ints_overlap()[:nbf, :nbf]
        # s_z is diagonal in the spin blocks, s_+ maps beta onto alpha
        S_z = np.zeros((2 * nbf, 2 * nbf), dtype=complex)
        S_z[:nbf, :nbf] = 0.5 * ovlp
        S_z[nbf:, nbf:] = -0.5 * ovlp
        S_plus = np.zeros((2 * nbf, 2 * nbf), dtype=complex)
        S_plus[:nbf, nbf:] = ovlp
    S_minus = S_plus.conj().T

    return tuple(C.conj().T @ A @ C for A in (S_z, S_plus, S_minus))


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
    S_z, S_plus, S_minus = spin_matrices(system, C, skip_picture_change)

    # S^2 = sum_i s_i.s_i + sum_{i!=j} s_i.s_j. 
    # s_i^2 = 3/4 for spin-1/2
    # This term is invariant under picture change
    spin2 = 0.75 * (ncore + np.trace(g1).real)

    # sum_{i!=j} s_i.s_j = - [(Sz)_ps (Sz)_qr + (S_-)_ps (S_+)_qr] g2[p,q,r,s], with
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
