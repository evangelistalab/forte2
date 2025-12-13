import numpy as np

from forte2 import ints
from forte2.integrals.libcint_utils import conc_env, basis_to_cint_envs

LIBCINT_AVAILABLE = getattr(ints, "HAS_LIBCINT", False)


def _require_libcint():
    if not LIBCINT_AVAILABLE:
        raise RuntimeError(
            "libcint integrals are unavailable. Rebuild forte2 with USE_LIBCINT=ON to enable libcint APIs."
        )


def _parse_basis_args_1e(system, basis1, basis2):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        basis1 = system.basis
        basis2 = system.basis
    elif basis1 is not None and basis2 is None:
        basis2 = basis1
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    return basis1, basis2


def _parse_basis_args_2c2e(system, basis1, basis2):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.auxiliary_basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        b1 = system.auxiliary_basis
        b2 = system.auxiliary_basis
    elif basis1 is not None and basis2 is None:
        b1 = basis1
        b2 = basis1
    elif basis1 is not None and basis2 is not None:
        b1 = basis1
        b2 = basis2
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    return b1, b2


def _parse_basis_args_3c2e(system, basis1, basis2, basis3):
    # basis1 defaults to system.auxiliary_basis if not provided
    # basis2 and 3 default to system.basis if not provided
    # if basis2 is provided and basis3 is not, basis3 = basis2
    # if both provided, use as is
    if basis1 is None:
        b1 = system.auxiliary_basis
    else:
        b1 = basis1

    if basis2 is None and basis3 is None:
        b2 = system.basis
        b3 = system.basis
    elif basis2 is not None and basis3 is None:
        b2 = basis2
        b3 = basis2
    elif basis2 is not None and basis3 is not None:
        b2 = basis2
        b3 = basis3
    elif basis2 is None and basis3 is not None:
        raise ValueError("If basis3 is provided, basis2 must also be provided.")

    return b1, b2, b3


def _parse_basis_args_4c2e(system, basis1, basis2, basis3, basis4):
    # 3 possible cases:
    # 1. all basis sets are None -> set all to system.basis
    # 2. basis1 is provided, others are None -> set others to basis1
    # 3. all basis sets are provided
    if basis1 is None and any(basis is not None for basis in [basis2, basis3, basis4]):
        raise ValueError(
            "If any of basis2, basis3, or basis4 is provided, basis1 must also be provided."
        )
    if basis1 is None:
        basis1 = basis2 = basis3 = basis4 = system.basis
    elif basis1 is not None and all(
        basis is None for basis in [basis2, basis3, basis4]
    ):
        basis2 = basis3 = basis4 = basis1
    return basis1, basis2, basis3, basis4


def nuclear_repulsion(system):
    r"""
    Compute the nuclear repulsion energy of the system.

    .. math::

        E_{nuc} = \frac{1}{2} \sum_{A \ne B} Z_A Z_B \iint
        \frac{\rho_A(\mathbf{r}_1) \rho_B(\mathbf{r}_2)}{r_{12}}
        d\mathbf{r}_1 d\mathbf{r}_2

    where :math:`Z_A` and :math:`Z_B` are the nuclear charges of atoms A and B,
    and :math:`\rho_A` and :math:`\rho_B` are their respective charge distributions.

    If point charges are used, then :math:`\rho_A=\delta(\mathbf{r} -
    \mathbf{R}_A)`, likewise for atom B, and the standard nuclear repulsion energy is computed.

    If Gaussian charges are used, then :math:`\rho_A=\left(\frac{\zeta_A}{\pi}\right)^{3/2}
    e^{-\zeta_A |\mathbf{r} - \mathbf{R}_A|^2}` where :math:`\zeta_A` is the Gaussian exponent for atom A, likewise for atom B.

    Parameters
    ----------
    system : System
        The molecular system containing the atomic charges and positions.

    Returns
    -------
    enuc : float
        The nuclear repulsion energy.
    """
    if system.use_gaussian_charges:
        ints2c = ints.coulomb_2c(
            system.gaussian_charge_basis, system.gaussian_charge_basis
        )
        ints2c -= np.diag(np.diag(ints2c))  # remove self-interaction terms
        enuc = 0.5 * np.einsum(
            "IJ,I,J->", ints2c, system.atomic_charges, system.atomic_charges
        )
        return enuc
    else:
        return ints.nuclear_repulsion(system.atoms)


def overlap(system, basis1=None, basis2=None):
    r"""
    Compute the overlap integral between two basis sets.

    .. math::

        S^{12}_{\mu\nu} = \int \chi^{1}_\mu(\mathbf{r}) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    return ints.overlap(basis1, basis2)


def kinetic(system, basis1=None, basis2=None):
    r"""
    Compute the kinetic energy integral between two basis sets.

    .. math::

        T^{12}_{\mu\nu} = -\frac{1}{2} \int \chi^{1}_\mu(\mathbf{r}) \nabla^2 \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    return ints.kinetic(basis1, basis2)


def nuclear(system, basis1=None, basis2=None):
    r"""
    Compute the nuclear attraction integral between two basis sets.

    .. math::

        V^{12}_{\mu\nu} = -\iint \chi^{1}_\mu(\mathbf{r}) \left(\sum_{A} \frac{Z_A \rho_A(|\mathbf{r}_A-\mathbf{R}_A|)}{|\mathbf{r} - \mathbf{R}_A|}\right) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r} d\mathbf{r}_A

    where :math:`Z_A` is the nuclear charge of atom A, and :math:`\rho_A` is its charge distribution.
    When point charges are used, :math:`\rho_A=\delta(\mathbf{r}_A - \mathbf{R}_A)`, and the standard nuclear attraction integral is recovered.
    When Gaussian charges are used, :math:`\rho_A=\left(\frac{\zeta_A}{\pi}\right)^{3/2} e^{-\zeta_A |\mathbf{r}_A - \mathbf{R}_A|^2}` where :math:`\zeta_A` is the Gaussian exponent for atom A, and the three-center, two-electron integral routine is used.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    if system.use_gaussian_charges:
        int3c = ints.coulomb_3c(system.gaussian_charge_basis, basis1, basis2)
        V = -np.einsum("Zpq,Z->pq", int3c, system.atomic_charges)
        return V
    else:
        return ints.nuclear(basis1, basis2, system.atoms)


def emultipole1(system, basis1=None, basis2=None, origin=None):
    r"""
    Compute the electric multipole moment integrals of up to first order (dipole)
    between two basis sets.
    Note that the zeroth order (overlap) is also returned as the first element.

    .. math::
        \mu^{12}_{\mu\nu,\alpha} = \int \chi^{1}_\mu(\mathbf{r}) r_\alpha \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    where :math:`\alpha` represents the x, y, or z component.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    origin : array-like, optional
        The origin for the multipole expansion. If None, defaults to [0.0, 0.0, 0.0].

    Returns
    -------
    emultipole1 : list[ndarray]
        The electric multipole moment integrals up to first order, in the order:
        [overlap, mu_x, mu_y, mu_z]
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    return ints.emultipole1(basis1, basis2, origin)


def emultipole2(system, basis1=None, basis2=None, origin=None):
    r"""
    Compute the electric multipole moment integrals of up to second order (quadrupole)
    between two basis sets.
    Note that the zeroth order (overlap) and first order (dipole) are also returned as the first four elements.

    .. math::
        Q^{12}_{\mu\nu,\alpha\beta} = \int \chi^{1}_\mu(\mathbf{r}) r_\alpha r_\beta \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    where :math:`\alpha` and :math:`\beta` represent the x, y, or z components.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    origin : array-like, optional
        The origin for the multipole expansion. If None, defaults to [0.0, 0.0, 0.0].

    Returns
    -------
    emultipole2 : list[ndarray]
        The electric multipole moment integrals up to second order, in the order:
        [overlap, mu_x, mu_y, mu_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz]
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    return ints.emultipole2(basis1, basis2, origin)


def emultipole3(system, basis1=None, basis2=None, origin=None):
    r"""
    Compute the electric multipole moment integrals of up to third order (octupole)
    between two basis sets.
    Note that the zeroth order (overlap), first order (dipole), and second order (quadrupole)
    are also returned as the first ten elements.

    .. math::
        O^{12}_{\mu\nu,\alpha\beta\gamma} = \int \chi^{1}_\mu(\mathbf{r}) r_\alpha r_\beta r_\gamma \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    where :math:`\alpha`, :math:`\beta`, and :math:`\gamma` represent the x, y, or z components.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    origin : array-like, optional
        The origin for the multipole expansion. If None, defaults to [0.0, 0.0, 0.0].

    Returns
    -------
    emultipole3 : list[ndarray]
        The electric multipole moment integrals up to third order, in the order:
        [overlap, mu_x, mu_y, mu_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz,
         O_xxx, O_xxy, O_xxz, O_xyy, O_xyz, O_xzz, O_yyy, O_yyz, O_yzz, O_zzz]
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    return ints.emultipole3(basis1, basis2, origin)


def opVop(system, basis1=None, basis2=None):
    r"""
    Compute the small component nuclear potential integral between two basis sets.

    .. math::
        W^{12}_{\mu\nu} = -\iint (\sigma\cdot\hat{p}) \chi^{1}_\mu(\mathbf{r}) \left(\sum_{A} \frac{Z_A \rho_A(|\mathbf{r}_A-\mathbf{R}_A|)}{|\mathbf{r} - \mathbf{R}_A|}\right) (\sigma\cdot\hat{p}) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r} d\mathbf{r}_A

    where :math:`Z_A` is the nuclear charge of atom A, and :math:`\rho_A` is its charge distribution.
    Currently, only point charges are supported for this integral.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.

    Returns
    -------
    opVop : list[ndarray]
        The small component nuclear potential integrals, in the order:
        [p dot Vp, (p cross Vp)_x, (p cross Vp)_y, (p cross Vp)_z]
    """
    # libint2 does not support 1e-opVop with Gaussian charges
    if system.use_gaussian_charges:
        # Requires libcint; raises a clear error if unavailable
        res = cint_opVop(system, basis1, basis2)
        # Note: libcint returns [sigma_x, sigma_y, sigma_z, I2].
        # We reorder to [I2, sigma_x, sigma_y, sigma_z] (the same as libint2)
        return [res[3], res[0], res[1], res[2]]
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    return ints.opVop(basis1, basis2, system.atoms)


def erf_nuclear(system, omega, basis1=None, basis2=None):
    r"""
    Compute the error function attenuated nuclear attraction integral between two basis sets.

    .. math::

        V^{12}_{\mu\nu}(\omega) = -\iint \chi^{1}_\mu(\mathbf{r}) \left(\sum_{A} \frac{Z_A \mathrm{erf}(\omega |\mathbf{r}-\mathbf{R}_A|)}{|\mathbf{r} - \mathbf{R}_A|}\right) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    where :math:`Z_A` is the nuclear charge of atom A, and :math:`\omega` is the attenuation parameter.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    omega : float
        The attenuation parameter for the error function.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.

    Returns
    -------
    V_erf : ndarray
        The error function attenuated nuclear attraction integral matrix.
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    return ints.erf_nuclear(basis1, basis2, (omega, system.atoms))


def erfc_nuclear(system, omega, basis1=None, basis2=None):
    r"""
    Compute the complementary error function attenuated nuclear attraction integral between two basis sets.

    .. math::

        V^{12}_{\mu\nu}(\omega) = -\iint \chi^{1}_\mu(\mathbf{r}) \left(\sum_{A} \frac{Z_A \mathrm{erfc}(\omega |\mathbf{r}-\mathbf{R}_A|)}{|\mathbf{r} - \mathbf{R}_A|}\right) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    where :math:`Z_A` is the nuclear charge of atom A, and :math:`\omega` is the attenuation parameter.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    omega : float
        The attenuation parameter for the complementary error function.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.

    Returns
    -------
    V_erfc : ndarray
        The complementary error function attenuated nuclear attraction integral matrix.
    """
    basis1, basis2 = _parse_basis_args_1e(system, basis1, basis2)
    return ints.erfc_nuclear(basis1, basis2, (omega, system.atoms))


def coulomb_4c(system, basis1=None, basis2=None, basis3=None, basis4=None):
    r"""
    Compute the four-center two-electron Coulomb integral between four basis sets.

    .. math::
        ( \mu\nu|\sigma\tau ) = \iint \chi^{1}_\mu(\mathbf{r}_1) \chi^{2}_\nu(\mathbf{r}_1)
        \frac{1}{r_{12}} \chi^{3}_\sigma(\mathbf{r}_2) \chi^{4}_\tau(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    basis3 : BasisSet, optional
        The third basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    basis4 : BasisSet, optional
        The fourth basis set. If None, defaults to system.basis or basis1 if basis1 is provided.

    Returns
    -------
    coulomb_4c : ndarray
        The four-center two-electron Coulomb integral tensor.
    """
    basis1, basis2, basis3, basis4 = _parse_basis_args_4c2e(
        system, basis1, basis2, basis3, basis4
    )
    return ints.coulomb_4c(basis1, basis2, basis3, basis4)


def coulomb_3c(system, basis1=None, basis2=None, basis3=None):
    r"""
    Compute the three-center two-electron Coulomb integral between three basis sets.

    .. math::
        ( P|\mu\nu ) = \iint \chi^{1}_P(\mathbf{r}_1) \frac{1}{r_{12}} \chi^{2}_\mu(\mathbf{r}_2) \chi^{3}_\nu(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.auxiliary_basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis.
    basis3 : BasisSet, optional
        The third basis set. If None, defaults to system.basis, or basis2 if basis2 is provided.

    Returns
    -------
    coulomb_3c : ndarray
        The three-center two-electron Coulomb integral tensor.
    """
    _basis1, _basis2, _basis3 = _parse_basis_args_3c2e(system, basis1, basis2, basis3)

    # max angular momentum supported:
    # libcint: 14
    # libint: 6
    max_l = max(_basis1.max_l, _basis2.max_l, _basis3.max_l)
    if max_l > 14:
        raise ValueError(
            f"coulomb_3c integral with basis functions of angular momentum > 14 "
            f"is not supported (max_l = {max_l})"
        )
    elif max_l > 6:
        return cint_coulomb_3c(system, basis1, basis2, basis3)
    else:
        return ints.coulomb_3c(_basis1, _basis2, _basis3)


def coulomb_2c(system, basis1=None, basis2=None):
    r"""
    Compute the two-center two-electron Coulomb integral between two basis sets.

    .. math::
        ( P|Q ) = \iint \chi^{1}_P(\mathbf{r}_1) \frac{1}{r_{12}} \chi^{2}_Q(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.auxiliary_basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.auxiliary_basis, or basis1 if basis1 is provided.

    Returns
    -------
    coulomb_2c : ndarray
        The two-center two-electron Coulomb integral matrix.
    """
    _basis1, _basis2 = _parse_basis_args_2c2e(system, basis1, basis2)

    # max angular momentum supported:
    # libcint: 14
    # libint: 6
    max_l = max(_basis1.max_l, _basis2.max_l)
    if max_l > 14:
        raise ValueError(
            f"coulomb_2c integral with basis functions of angular momentum > 14 "
            f"is not supported (max_l = {max_l})"
        )
    elif max_l > 6:
        return cint_coulomb_2c(system, basis1, basis2)
    else:
        return ints.coulomb_2c(_basis1, _basis2)


def erf_coulomb_3c(system, omega, basis1=None, basis2=None, basis3=None):
    r"""
    Compute the error function attenuated three-center two-electron Coulomb integral between three basis sets.

    .. math::
        ( P|\mu\nu )(\omega) = \iint \chi^{1}_P(\mathbf{r}_1) \frac{\mathrm{erf}(\omega r_{12})}{r_{12}} \chi^{2}_\mu(\mathbf{r}_2) \chi^{3}_\nu(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    where :math:`\omega` is the attenuation parameter.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    omega : float
        The attenuation parameter for the error function.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.auxiliary_basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis.
    basis3 : BasisSet, optional
        The third basis set. If None, defaults to system.basis, or basis2 if basis2 is provided.

    Returns
    -------
    erf_coulomb_3c : ndarray
        The error function attenuated three-center two-electron Coulomb integral tensor.
    """
    basis1, basis2, basis3 = _parse_basis_args_3c2e(system, basis1, basis2, basis3)
    return ints.erf_coulomb_3c(basis1, basis2, basis3, omega)


def erf_coulomb_2c(system, omega, basis1=None, basis2=None):
    r"""
    Compute the error function attenuated two-center two-electron Coulomb integral between two basis sets.

    .. math::
        ( P|Q )(\omega) = \iint \chi^{1}_P(\mathbf{r}_1) \frac{\mathrm{erf}(\omega r_{12})}{r_{12}} \chi^{2}_Q(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    where :math:`\omega` is the attenuation parameter.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    omega : float
        The attenuation parameter for the error function.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.auxiliary_basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.auxiliary_basis, or basis1 if basis1 is provided.

    Returns
    -------
    erf_coulomb_2c : ndarray
        The error function attenuated two-center two-electron Coulomb integral matrix.
    """
    basis1, basis2 = _parse_basis_args_2c2e(system, basis1, basis2)
    return ints.erf_coulomb_2c(basis1, basis2, omega)


def erfc_coulomb_3c(system, omega, basis1=None, basis2=None, basis3=None):
    r"""
    Compute the complementary error function attenuated three-center two-electron Coulomb integral between three basis sets.

    .. math::
        ( P|\mu\nu )(\omega) = \iint \chi^{1}_P(\mathbf{r}_1) \frac{\mathrm{erfc}(\omega r_{12})}{r_{12}} \chi^{2}_\mu(\mathbf{r}_2) \chi^{3}_\nu(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    where :math:`\omega` is the attenuation parameter.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    omega : float
        The attenuation parameter for the complementary error function.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.auxiliary_basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis.
    basis3 : BasisSet, optional
        The third basis set. If None, defaults to system.basis, or basis2 if basis2 is provided.

    Returns
    -------
    erfc_coulomb_3c : ndarray
        The complementary error function attenuated three-center two-electron Coulomb integral tensor.
    """
    basis1, basis2, basis3 = _parse_basis_args_3c2e(system, basis1, basis2, basis3)
    return ints.erfc_coulomb_3c(basis1, basis2, basis3, omega)


def erfc_coulomb_2c(system, omega, basis1=None, basis2=None):
    r"""
    Compute the complementary error function attenuated two-center two-electron Coulomb integral between two basis sets.

    .. math::
        ( P|Q )(\omega) = \iint \chi^{1}_P(\mathbf{r}_1) \frac{\mathrm{erfc}(\omega r_{12})}{r_{12}} \chi^{2}_Q(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    where :math:`\omega` is the attenuation parameter.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    omega : float
        The attenuation parameter for the complementary error function.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.auxiliary_basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.auxiliary_basis, or basis1 if basis1 is provided.

    Returns
    -------
    erfc_coulomb_2c : ndarray
        The complementary error function attenuated two-center two-electron Coulomb integral matrix.
    """
    basis1, basis2 = _parse_basis_args_2c2e(system, basis1, basis2)
    return ints.erfc_coulomb_2c(basis1, basis2, omega)


def _parse_basis_args_cint_1e(system, basis1, basis2, origin=None):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        atm, bas, env = basis_to_cint_envs(system, system.basis, common_origin=origin)
        shell_slice = [0, system.basis.nshells, 0, system.basis.nshells]
    elif basis1 is not None and basis2 is None:
        atm, bas, env = basis_to_cint_envs(system, basis1, common_origin=origin)
        shell_slice = [0, basis1.nshells, 0, basis1.nshells]
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    else:
        atm1, bas1, env1 = basis_to_cint_envs(system, basis1, common_origin=origin)
        atm2, bas2, env2 = basis_to_cint_envs(system, basis2, common_origin=origin)
        atm, bas, env = conc_env(atm1, bas1, env1, atm2, bas2, env2)
        ns1 = basis1.nshells
        ns2 = basis2.nshells
        shell_slice = [0, ns1, ns1, ns1 + ns2]
    return atm, bas, env, shell_slice


def _parse_basis_args_cint_2c2e(system, basis1, basis2, origin=None):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.auxiliary_basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        atm, bas, env = basis_to_cint_envs(
            system, system.auxiliary_basis, common_origin=origin
        )
        shell_slice = [
            0,
            system.auxiliary_basis.nshells,
            0,
            system.auxiliary_basis.nshells,
        ]
    elif basis1 is not None and basis2 is None:
        atm, bas, env = basis_to_cint_envs(system, basis1, common_origin=origin)
        shell_slice = [0, basis1.nshells, 0, basis1.nshells]
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    else:
        atm1, bas1, env1 = basis_to_cint_envs(system, basis1, common_origin=origin)
        atm2, bas2, env2 = basis_to_cint_envs(system, basis2, common_origin=origin)
        atm, bas, env = conc_env(atm1, bas1, env1, atm2, bas2, env2)
        ns1 = basis1.nshells
        ns2 = basis2.nshells
        shell_slice = [0, ns1, ns1, ns1 + ns2]
    return atm, bas, env, shell_slice


def _parse_basis_args_cint_3c2e(system, basis1, basis2, basis3, origin=None):
    # Note that cint expects (ij | P), but we output in (P | ij) layout like libint2.
    # We handle all 8 possible cases for basis set inputs
    if basis1 is None:
        aux_atm, aux_bas, aux_env = basis_to_cint_envs(
            system, system.auxiliary_basis, common_origin=origin
        )
        nsh_aux = system.auxiliary_basis.nshells
    else:
        aux_atm, aux_bas, aux_env = basis_to_cint_envs(
            system, basis1, common_origin=origin
        )
        nsh_aux = basis1.nshells

    if basis2 is None and basis3 is None:
        bas_atm, bas_bas, bas_env = basis_to_cint_envs(
            system, system.basis, common_origin=origin
        )
        nsh_bas = system.basis.nshells
        shell_slice = [0, nsh_bas, 0, nsh_bas, nsh_bas, nsh_bas + nsh_aux]
    elif basis2 is not None and basis3 is None:
        bas_atm, bas_bas, bas_env = basis_to_cint_envs(
            system, basis2, common_origin=origin
        )
        nsh_bas = basis2.nshells
        shell_slice = [0, nsh_bas, 0, nsh_bas, nsh_bas, nsh_bas + nsh_aux]
    elif basis2 is not None and basis3 is not None and basis2 != basis3:
        raise ValueError(
            "libcint doesn't support (P|QR) with Q and R being different basis sets."
        )
    else:
        raise ValueError("If basis3 is provided, basis2 must also be provided.")

    atm, bas, env = conc_env(bas_atm, bas_bas, bas_env, aux_atm, aux_bas, aux_env)
    return atm, bas, env, shell_slice


def _f2c(arr):
    if arr.shape[-1] == 1:
        return np.ascontiguousarray(arr[..., 0])
    else:
        return np.ascontiguousarray(np.rollaxis(arr, -1, 0))


def cint_overlap(system, basis1=None, basis2=None):
    r"""
    Compute the overlap integral between two basis sets using the Libcint library.

    .. math::

        S^{12}_{\mu\nu} = \int \chi^{1}_\mu(\mathbf{r}) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    res = ints.cint_int1e_ovlp_sph(shell_slice, atm, bas, env)
    return _f2c(res)


def cint_overlap_spinor(system, basis1=None, basis2=None):
    r"""
    Compute the overlap integral between two basis sets using the Libcint library.

    .. math::

        S^{12}_{\mu\nu} = \int \chi^{1}_\mu(\mathbf{r}) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    res = ints.cint_int1e_ovlp_spinor(shell_slice, atm, bas, env)
    return _f2c(res)


def cint_kinetic(system, basis1=None, basis2=None):
    r"""
    Compute the kinetic energy integral between two basis sets using the Libcint library.

    .. math::

        T^{12}_{\mu\nu} = -\frac{1}{2} \int \chi^{1}_\mu(\mathbf{r}) \nabla^2 \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    res = ints.cint_int1e_kin_sph(shell_slice, atm, bas, env)
    return _f2c(res)


def cint_nuclear(system, basis1=None, basis2=None):
    r"""
    Compute the nuclear attraction integral between two basis sets using the Libcint library.

    .. math::

        V^{12}_{\mu\nu} = -\iint \chi^{1}_\mu(\mathbf{r}) \left(\sum_{A} \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}\right) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    res = ints.cint_int1e_nuc_sph(shell_slice, atm, bas, env)
    return _f2c(res)


def cint_opVop(system, basis1=None, basis2=None):
    r"""
    Compute the small component nuclear potential integral between two basis sets using the Libcint library.

    .. math::
        W^{12}_{\mu\nu} = -\iint (\sigma\cdot\hat{p}) \chi^{1}_\mu(\mathbf{r}) \left(\sum_{A} \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}\right) (\sigma\cdot\hat{p}) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.

    Returns
    -------
    opVop : ndarray
        The small component nuclear potential integrals. Order of components:
        [(p cross Vp)_x, (p cross Vp)_y, (p cross Vp)_z, p dot Vp]
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    res = ints.cint_int1e_spnucsp_sph(shell_slice, atm, bas, env)
    # C-layout, first index is the integral component (slowest changing)
    return _f2c(res)


def cint_opVop_spinor(system, basis1=None, basis2=None):
    r"""
    Compute the small component nuclear potential integral between two basis sets using the Libcint library.

    .. math::
        W^{12}_{\mu\nu} = -\iint (\sigma\cdot\hat{p}) \chi^{1}_\mu(\mathbf{r}) \left(\sum_{A} \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|}\right) (\sigma\cdot\hat{p}) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.

    Returns
    -------
    opVop_spinor : ndarray
        The small component nuclear potential integrals in spinor basis
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    res = ints.cint_int1e_spnucsp_spinor(shell_slice, atm, bas, env)
    return _f2c(res)


def cint_emultipole1(system, basis1=None, basis2=None, origin=None):
    r"""
    Compute the electric multipole moment integrals of up to first order (dipole)
    between two basis sets using the Libcint library.
    Note that the zeroth order (overlap) is also returned as the first element.

    .. math::
        \mu^{12}_{\mu\nu,\alpha} = \int \chi^{1}_\mu(\mathbf{r}) r_\alpha \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    where :math:`\alpha` represents the x, y, or z component.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    origin : array-like, optional
        The origin for the multipole expansion. If None, defaults to [0.0, 0.0, 0.0].

    Returns
    -------
    emultipole1 : ndarray
        The first electric multipole moment integrals
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(
        system, basis1, basis2, origin
    )
    res = ints.cint_int1e_r_sph(shell_slice, atm, bas, env)
    # C-layout, first index is the integral component (slowest changing)
    return _f2c(res)


def cint_sprsp(system, basis1=None, basis2=None, origin=None):
    r"""
    Compute the small-component dipole integral between two basis sets using the Libcint library.

    .. math::
        D^{12}_{\mu\nu,\alpha} = \int (\sigma\cdot\hat{p}) \chi^{1}_\mu(\mathbf{r}) r_\alpha (\sigma\cdot\hat{p}) \chi^{2}_\nu(\mathbf{r}) d\mathbf{r}

    where :math:`\alpha` represents the x, y, or z component.

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.basis or basis1 if basis1 is provided.
    origin : array-like, optional
        The origin for the multipole expansion. If None, defaults to [0.0, 0.0, 0.0].

    Returns
    -------
    sprsp : ndarray
        The small-component dipole integrals. Order of components:
        [ mu_{x, sigma_x}, mu_{x, sigma_y}, mu_{x, sigma_z}, mu_{x, I2},
          mu_{y, sigma_x}, mu_{y, sigma_y}, mu_{y, sigma_z}, mu_{y, I2},
          mu_{z, sigma_x}, mu_{z, sigma_y}, mu_{z, sigma_z}, mu_{z, I2} ]
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(
        system, basis1, basis2, origin
    )
    res = ints.cint_int1e_sprsp_sph(shell_slice, atm, bas, env)
    # C-layout, first index is the integral component (slowest changing)
    return _f2c(res)


def cint_coulomb_2c(system, basis1=None, basis2=None):
    r"""
    Compute the two-center two-electron Coulomb integral between two basis sets using the Libcint library.

    .. math::
        ( P|Q ) = \iint \chi^{1}_P(\mathbf{r}_1) \frac{1}{r_{12}} \chi^{2}_Q(\mathbf{r}_2)
        d\mathbf{r}_1 d\mathbf{r}_2

    Parameters
    ----------
    system : System
        The molecular system containing the basis sets.
    basis1 : BasisSet, optional
        The first basis set. If None, defaults to system.auxiliary_basis.
    basis2 : BasisSet, optional
        The second basis set. If None, defaults to system.auxiliary_basis, or basis1 if basis1 is provided.

    Returns
    -------
    ndarray
        The two-center two-electron Coulomb integral matrix.
    """
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_2c2e(system, basis1, basis2)
    res = ints.cint_int2c2e_sph(shell_slice, atm, bas, env)
    return _f2c(res)


def cint_coulomb_3c(system, basis1=None, basis2=None, basis3=None):
    _require_libcint()
    atm, bas, env, shell_slice = _parse_basis_args_cint_3c2e(
        system, basis1, basis2, basis3
    )
    res = ints.cint_int3c2e_sph(shell_slice, atm, bas, env)
    return res.transpose(2, 0, 1).copy()  # copy makes sure it's C-contiguous
