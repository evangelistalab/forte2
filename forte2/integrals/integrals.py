import numpy as np

from forte2 import ints
from forte2.integrals.libcint_utils import conc_env


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
        basis1 = system.auxiliary_basis
        basis2 = system.auxiliary_basis
    elif basis1 is not None and basis2 is None:
        basis2 = basis1
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    return basis1, basis2


def _parse_basis_args_3c2e(system, basis1, basis2, basis3):
    # 3 possible cases:
    # 1. all basis sets are None -> set basis1 to system.auxiliary, basis2 and basis3 to system.basis
    # 2. basis1 is provided, basis2 and basis3 are None -> set basis2 and basis3 to system.basis
    # 3. all basis sets are provided
    if basis1 is None:
        basis1 = system.auxiliary_basis
    if basis2 is None and basis3 is not None:
        raise ValueError("If basis3 is provided, basis2 must also be provided.")
    if basis2 is None:
        basis2 = system.basis
    if basis3 is None:
        basis3 = system.basis

    return basis1, basis2, basis3


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
        [p dot Vp, (p cross Vp)_z, (p cross Vp)_x, (p cross Vp)_y]
    """
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
    basis1, basis2, basis3 = _parse_basis_args_3c2e(system, basis1, basis2, basis3)
    return ints.coulomb_3c(basis1, basis2, basis3)


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
    basis1, basis2 = _parse_basis_args_2c2e(system, basis1, basis2)
    return ints.coulomb_2c(basis1, basis2)


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


def _parse_basis_args_cint_1e(system, basis1, basis2):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        atm = system.basis.cint_atm
        bas = system.basis.cint_bas
        env = system.basis.cint_env
        shell_slice = [0, system.basis.nshells, 0, system.basis.nshells]
    elif basis1 is not None and basis2 is None:
        atm = basis1.cint_atm
        bas = basis1.cint_bas
        env = basis1.cint_env
        shell_slice = [0, basis1.nshells, 0, basis1.nshells]
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    else:
        atm, bas, env = conc_env(
            basis1.cint_atm,
            basis1.cint_bas,
            basis1.cint_env,
            basis2.cint_atm,
            basis2.cint_bas,
            basis2.cint_env,
        )
        ns1 = basis1.nshells
        ns2 = basis2.nshells
        shell_slice = [0, ns1, ns1, ns1 + ns2]
    return atm, bas, env, shell_slice


def _parse_basis_args_cint_2c2e(system, basis1, basis2):
    # 2 possible cases:
    # 1. both basis sets are None -> set both to system.auxiliary_basis
    # 2. basis1 is provided, basis2 is None -> set basis2 to basis1
    if basis1 is None and basis2 is None:
        atm = system.auxiliary_basis.cint_atm
        bas = system.auxiliary_basis.cint_bas
        env = system.auxiliary_basis.cint_env
        shell_slice = [
            0,
            system.auxiliary_basis.nshells,
            0,
            system.auxiliary_basis.nshells,
        ]
    elif basis1 is not None and basis2 is None:
        atm = basis1.cint_atm
        bas = basis1.cint_bas
        env = basis1.cint_env
        shell_slice = [0, basis1.nshells, 0, basis1.nshells]
    elif basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    else:
        atm, bas, env = conc_env(
            basis1.cint_atm,
            basis1.cint_bas,
            basis1.cint_env,
            basis2.cint_atm,
            basis2.cint_bas,
            basis2.cint_env,
        )
        ns1 = basis1.nshells
        ns2 = basis2.nshells
        shell_slice = [0, ns1, ns1, ns1 + ns2]
    return atm, bas, env, shell_slice


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
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    return ints.cint_int1e_ovlp_sph(shell_slice, atm, bas, env)


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
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    return ints.cint_int1e_ovlp_spinor(shell_slice, atm, bas, env)


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
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    return ints.cint_int1e_kin_sph(shell_slice, atm, bas, env)


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
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    return ints.cint_int1e_nuc_sph(shell_slice, atm, bas, env)

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
        The small component nuclear potential integrals.
    """
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    return ints.cint_int1e_spnucsp_sph(shell_slice, atm, bas, env)

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
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    return ints.cint_int1e_spnucsp_spinor(shell_slice, atm, bas, env)

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
    atm, bas, env, shell_slice = _parse_basis_args_cint_1e(system, basis1, basis2)
    # if origin is None:
    #     origin = [0.0, 0.0, 0.0]
    return ints.cint_int1e_r_sph(shell_slice, atm, bas, env)

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
    atm, bas, env, shell_slice = _parse_basis_args_cint_2c2e(system, basis1, basis2)
    return ints.cint_int2c2e_sph(shell_slice, atm, bas, env)
