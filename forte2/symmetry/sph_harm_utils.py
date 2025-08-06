import numpy as np
import scipy as sp


def sph_real_to_complex(l):
    """
    Conversion matrix from real spherical harmonics to complex spherical harmonics for a given l.

    Parameters
    ----------
    l : int
        The angular momentum quantum number.

    Returns
    -------
    NDArray
        The conversion matrix of shape (2*l+1, 2*l+1).

    Notes
    -----
    See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    Y^m_l are the complex spherical harmonics, and Y_{lm} are the real spherical harmonics.
    """
    nml = 2 * l + 1
    U = np.zeros((nml, nml), dtype=np.complex128)

    # Y^0_l = Y_{l0}
    U[l, l] = 1.0
    for m in range(1, l + 1):
        # Condon-Shortley phase
        p = 1 if m % 2 == 0 else -1
        # m < 0 case
        U[l - m, l - m] = p * (-1.0j / np.sqrt(2.0))
        U[l - m, l + m] = p * (1.0 / np.sqrt(2.0))
        # m > 0 case
        U[l + m, l - m] = 1.0j / np.sqrt(2.0)
        U[l + m, l + m] = 1.0 / np.sqrt(2.0)

    return U


def sph_complex_to_real(l):
    """
    Conversion matrix from complex spherical harmonics to real spherical harmonics for a given l.

    Parameters
    ----------
    l : int
        The angular momentum quantum number.

    Returns
    -------
    NDArray
        The conversion matrix of shape (2*l+1, 2*l+1).
    """
    return sph_real_to_complex(l).T.conj()


def clebsh_gordan_spin_half(l, msdouble, jdouble, mjdouble):
    r"""
    Clebsch Gordon coefficient specialized for coupling orbital angular momentum l
    with spin 1/2 to form total angular momentum j = l + 1/2, |l - 1/2|

    .. math::
        C^{j,m_j}_{l,m_l;1/2,m_s} = \langle l,m_l;\frac{1}{2},m_s|j,m_j\rangle

    Since m_s + m_l must equal m_j, specifying m_j and m_s uniquely determines m_l.

    Parameters
    ----------
    l : int
        The orbital angular momentum quantum number.
    jdouble : int
        The total angular momentum quantum number
        j is either l + 1/2 or l - 1/2.
    mjdouble : int
        The magnetic quantum number of the total angular momentum.
        m_j can take a value in {j, j-1, ..., -j}.
    msdouble : int
        The magnetic quantum number of the z-component of the total angular momentum.
        mz is either 1/2 or -1/2.

    Notes
    -----
    1. Ch. 4.1.2, Atkins, P. W. (2011). Molecular Quantum Mechanics Fifth Ed.
    2. https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients#Special_cases
    """
    # TODO: asserts?
    nml = 2 * l + 1
    if jdouble == 2 * l + 1:
        if msdouble == 1:
            c = np.sqrt(0.5 * (nml + mjdouble) / nml)
        elif msdouble == -1:
            c = np.sqrt(0.5 * (nml - mjdouble) / nml)
    elif jdouble == 2 * l - 1:
        if msdouble == 1:
            c = -np.sqrt(0.5 * (nml - mjdouble) / nml)
        elif msdouble == -1:
            c = np.sqrt(0.5 * (nml + mjdouble) / nml)
    else:
        c = 0
    return c


def real_sph_to_j_adapted_per_l(l):
    r"""
    Transformation matrix that transforms real spherical harmonics 
    of a given angular momentum to (j-adapted) spinor basis.

    Parameters
    ----------
    l : int
        The angular momentum quantum number.

    Returns
    -------
    tuple[NDArray]
        The transformation matrices for alpha and beta spinors.
        Each matrix has shape (2*l+1, 4*l+2).

    Notes
    -----
    The transformation is based on the Clebsch-Gordan coefficients for coupling
    the orbital angular momentum l with spin 1/2 to form total angular momentum j = l + 1/2, l - 1/2.
    l = 0 is a special case where the transformation is trivial.

    .. math::
        |j, m_j> = \sum_{m_l=-l}^l\sum_{m_s=-1/2}^{1/2} |l, m_l; 1/2, m_s\rangle\langle l, m_l; 1/2, m_s|j, m_j\rangle\\
                = \sum_{m_l=-l}^l\sum_{m_s=-1/2}^{1/2} |l, m_l; 1/2, m_s\rangle * C^{j,m_j}_{l,m_l;1/2,m_s}.

    """
    if l == 0:
        # mj = -1/2: 'beta', mj = 1/2: 'alpha', hence the order
        return np.array((0.0, 1.0)).reshape(1, -1), np.array((1.0, 0.0)).reshape(1, -1)

    r2c = sph_real_to_complex(l)
    nml = 2 * l + 1
    nmj = 4 * l + 2
    ua = np.zeros((nml, nmj), dtype=np.complex128)
    ub = np.zeros((nml, nmj), dtype=np.complex128)

    # 'ua/b' first transforms the real spherical AOs to complex spherical AOs
    # (pure angular momentum eigenfunctions), i.e., R(r)*|l, m_l>.
    # 'ua' then applies the Clebsch-Gordan coefficients for m_s=1/2,
    # and 'ub' for m_s=-1/2.
    # 'ua' and 'ub' have dimensions (2*l+1, 4*l+2):
    # | j= l-1/2         j = l+1/2       |
    # | (2l+1) * (2l)    (2l+1) * (2l+2) |

    # Case 1: j = l-1/2, m_j goes from -(l-1/2) to l-1/2
    # since m_l + m_s = m_j,
    # so for alpha, the lowest ml is -(l-1/2)-1/2 = -l
    # and for beta, the lowest ml is -(l-1/2)+1/2 = -l+1
    # these correspond to the r2c[:, 0] and r2c[:, 1] columns
    jdouble = l * 2 - 1
    mldouble_a = 0
    mldouble_b = 1
    for k, mjdouble in enumerate(range(-jdouble, jdouble + 1, 2)):
        ua[:, k] = r2c[:, mldouble_a] * clebsh_gordan_spin_half(l, 1, jdouble, mjdouble)
        ub[:, k] = r2c[:, mldouble_b] * clebsh_gordan_spin_half(
            l, -1, jdouble, mjdouble
        )
        mldouble_a += 1
        mldouble_b += 1

    # Case 2: j = l + 1/2
    jdouble = l * 2 + 1
    mldouble_a = -1
    mldouble_b = 0
    for k, mjdouble in enumerate(range(-jdouble, jdouble + 1, 2)):
        if mldouble_a < 0:
            # corresponds to m_l = -l-1 (invalid)
            ua[:, l * 2 + k] = 0
        else:
            ua[:, l * 2 + k] = r2c[:, mldouble_a] * clebsh_gordan_spin_half(
                l, 1, jdouble, mjdouble
            )
        if mldouble_b >= 2 * l + 1:
            # corresponds to m_l = l+1 (invalid)
            ub[:, l * 2 + k] = 0
        else:
            ub[:, l * 2 + k] = r2c[:, mldouble_b] * clebsh_gordan_spin_half(
                l, -1, jdouble, mjdouble
            )
        mldouble_a += 1
        mldouble_b += 1
    return ua, ub


def real_sph_to_j_adapted(basis):
    """
    Transformation matrix that transforms real-spherical GTOs to spinor
    GTOs for all basis functions
    """
    # get transformation matrices for each l
    lmax = basis.max_l
    ualst = []
    ublst = []
    for l in range(lmax + 1):
        ua, ub = real_sph_to_j_adapted_per_l(l)
        ualst.append(ua)
        ublst.append(ub)

    ca = []
    cb = []
    for ishell in range(basis.nshells):
        l = basis[ishell].l
        ua = ualst[l]
        ub = ublst[l]
        nctr = basis[ishell].ncontr
        ca.extend([ua] * nctr)
        cb.extend([ub] * nctr)
    return sp.linalg.block_diag(*ca), sp.linalg.block_diag(*cb)
