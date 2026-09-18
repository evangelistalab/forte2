import numpy as np

from forte2 import integrals
from forte2.data.atom_data import LIGHT_SPEED
from forte2.state import StateAverageInfo
from forte2.helpers import logger, block_diag_2x2, sigma_dot


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


def magnetic_moment_matrices(system, C, origin=None, skip_picture_change=None):
    r"""
    Build the magnetic dipole moment matrices in the spinor basis spanned by `C`.

    Parameters
    ----------
    system : System
        The two-component system.
    C : NDArray
        The spinor coefficients, shape (2*nbf, nspinor).
    origin : array-like, optional
        The gauge origin. If None, defaults to [0, 0, 0]. The orbital contribution to the
        magnetic moment is gauge-origin dependent, so this is part of the definition of
        the property.
    skip_picture_change : bool | None, optional, default=None
        If True, use the nonrelativistic form :math:`-(\hat{L} + 2\hat{S})/2c` instead of
        the picture-change-corrected operator. If None, follow ``skip_picture_change`` on
        the system's X2C parameters.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The :math:`\hat{m}_x`, :math:`\hat{m}_y`, :math:`\hat{m}_z` matrices, each of
        shape (nspinor, nspinor), in atomic units.
    """
    nbf = system.nbf
    assert (
        C.shape[0] == 2 * nbf
    ), "C must be in the spinorbital basis, with shape (2*nbf, nspinor)."
    if skip_picture_change is None:
        skip_picture_change = system.skip_picture_change

    if system.x2c_type in ["sf", "so"] and not skip_picture_change:
        m = system.x2c_helper.magnetic_dipole_moment(origin=origin)
    else:
        ovlp = system.ints_overlap()[:nbf, :nbf]
        # L = -i (r x nabla); the Bohr magneton is 1/2c in these units
        lmat = integrals.cint_cg_irxp(system, origin=origin)
        zero = np.zeros_like(ovlp)
        spin = [
            0.5 * sigma_dot(ovlp, zero, zero),
            0.5 * sigma_dot(zero, ovlp, zero),
            0.5 * sigma_dot(zero, zero, ovlp),
        ]
        m = [
            -(block_diag_2x2(-1j * lmat[k]) + 2 * spin[k]) / (2 * LIGHT_SPEED)
            for k in range(3)
        ]

    return tuple(C.conj().T @ mk @ C for mk in m)


def compute_g_tensor(
    system, C, g1_aa, g1_ab, g1_bb, origin=None, skip_picture_change=None
):
    r"""
    Compute the g-tensor of a Kramers doublet from the magnetic dipole moment operator.

    Parameters
    ----------
    system : System
        The two-component system.
    C : NDArray
        The coefficients of the core and active spinors, shape (2*nbf, ncore + nactv),
        with the core columns first. The number of core spinors follows from the size of
        `g1_aa`.
    g1_aa, g1_bb : NDArray
        The active-space one-particle density matrices of the two doublet components.
    g1_ab : NDArray
        The active-space one-particle transition density matrix between them,
        :math:`\gamma_{pq} = \langle \Psi_a | a^\dagger_p a_q | \Psi_b \rangle`.
    origin : array-like, optional
        The gauge origin, see :func:`magnetic_moment_matrices`.
    skip_picture_change : bool | None, optional, default=None
        See :func:`magnetic_moment_matrices`.

    Returns
    -------
    g_values : NDArray
        The three principal g-values, in ascending order.
    g_axes : NDArray
        The corresponding principal axes as columns, shape (3, 3).

    Notes
    -----
    Within a Kramers doublet the Zeeman interaction is represented by the effective
    Hamiltonian :math:`\hat{H} = \mu_B \mathbf{B}\cdot g\cdot \tilde{S}` for a pseudospin
    :math:`\tilde{S} = 1/2`, so the moment operator is
    :math:`\hat{m}_k = -\mu_B \sum_l g_{kl} \tilde{S}_l`. Taking traces over the
    two-dimensional model space gives

    .. math::
        \mathrm{Tr}[M_k M_l] = \frac{\mu_B^2}{2} (g g^T)_{kl}

    so the principal g-values are the square roots of the eigenvalues of
    :math:`(2/\mu_B^2)\,\mathrm{Tr}[M_k M_l]`, with :math:`\mu_B = 1/2c`. Because only
    traces enter, the result does not depend on which orthonormal basis of the doublet
    the two states happen to be reported in, which is what makes this definition
    well-posed. Signs of the principal values are not determined this way.

    The accuracy is limited by the restricted kinetic balance used for the
    picture-change correction of the moment operator; see
    :meth:`~forte2.x2c.x2c.X2CHelper.magnetic_dipole_moment`.
    """
    ncore = C.shape[1] - g1_aa.shape[0]
    m = magnetic_moment_matrices(system, C, origin, skip_picture_change)

    # the 2x2 matrix of each moment component within the doublet
    M = np.zeros((3, 2, 2), dtype=complex)
    for k in range(3):
        m_cc, _, _, m_aa = _split_blocks(m[k], ncore)
        core = np.trace(m_cc)
        M[k, 0, 0] = core + np.einsum("uv,uv->", m_aa, g1_aa)
        M[k, 1, 1] = core + np.einsum("uv,uv->", m_aa, g1_bb)
        # the off-diagonal block carries no core contribution: the two states are
        # orthogonal and share the same core
        M[k, 0, 1] = np.einsum("uv,uv->", m_aa, g1_ab)
        M[k, 1, 0] = M[k, 0, 1].conj()

    # mu_B = 1/2c, so 2/mu_B^2 = 8 c^2
    A = 8 * LIGHT_SPEED**2 * np.einsum("kab,lba->kl", M, M).real
    evals, g_axes = np.linalg.eigh(A)
    g_values = np.sqrt(np.clip(evals, 0.0, None))
    return g_values, g_axes


def pretty_print_g_tensor(g_values, g_axes, roots, header="\nKramers doublet g-tensor"):
    r"""
    Print the principal g-values and axes of a Kramers doublet.

    Parameters
    ----------
    g_values : NDArray
        The three principal g-values.
    g_axes : NDArray
        The principal axes as columns, shape (3, 3).
    roots : tuple[int, int]
        The two CI roots forming the doublet.
    header : str, optional
        A header string to display at the top of the summary.
    """
    logger.log_info1(f"{header} (roots {roots[0]} and {roots[1]}):")
    width = 52
    logger.log_info1("=" * width)
    logger.log_info1(f"{'':>10} {'g':>12} {'x':>8} {'y':>8} {'z':>8}")
    logger.log_info1("-" * width)
    for i in range(3):
        ax = g_axes[:, i]
        logger.log_info1(
            f"{'g_' + str(i + 1):>10} {g_values[i]:>12.6f} "
            f"{ax[0]:>8.4f} {ax[1]:>8.4f} {ax[2]:>8.4f}"
        )
    logger.log_info1("-" * width)
    logger.log_info1(f"{'g_iso':>10} {np.mean(g_values):>12.6f}")
    logger.log_info1("=" * width)


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
