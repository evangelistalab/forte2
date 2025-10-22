import numpy as np
import scipy as sp

from forte2 import ints, Basis, Shell
from forte2.system import System
from forte2.system.build_basis import build_basis
from forte2.helpers.matrix_functions import givens_rotation


def minao_initial_guess(system, H):
    """
    Generate a superposition of atomic potentials (SAP) initial guess for the SCF procedure
    S. Lehtola, J. Chem. Theory Comput. 15, 1593-1604 (2019), arXiv:1810.11659.
    For details, see https://doi.org/10.1063/5.0004046

    Parameters
    ----------
    system : forte2.System
        The system object containing the atoms and basis set.
    H : NDArray
        The Fock matrix.
    S : NDArray
        The overlap matrix.

    Returns
    -------
    NDArray
        The initial MO guess for the SCF procedure.
    """

    # generate the SAP basis from the initial guess file. Skip normalization
    sap_basis, _ = build_basis(
        "sap_helfem_large",
        system.geom_helper,
        embed_normalization_into_coefficients=False,
    )

    # create a new basis that will be used to store the scaled coefficients
    scaled_sap_basis = Basis()

    for shell in sap_basis:
        # scales the coefficients by -(exponent / pi)^(3/2)
        scaled_coeff = np.array(
            [-c * ((e / np.pi) ** 1.5) for c, e in zip(shell.coeff, shell.exponents)]
        )
        scaled_shell = Shell(
            shell.l,
            shell.exponents,
            scaled_coeff,
            shell.center,
            shell.is_pure,
            embed_normalization_into_coefficients=False,  # do not normalize
        )
        scaled_sap_basis.add(scaled_shell)

    # generate the SAP integrals (P|mn)
    SAP_ints = ints.coulomb_3c(scaled_sap_basis, system.basis, system.basis)

    # generate the SAP potential V_mn = sum_P (P|mn)
    SAP_V = np.einsum("Pmn->mn", SAP_ints)

    if system.two_component:
        _SAP_V = sp.linalg.block_diag(SAP_V, SAP_V).astype(complex)
    else:
        _SAP_V = SAP_V

    # generate the SAP Hamiltonian and diagonalize it
    Xorth = system.get_Xorth()
    H_SAP = Xorth.T @ (H + _SAP_V) @ Xorth
    _, C = np.linalg.eigh(H_SAP)

    return Xorth @ C


def core_initial_guess(system: System, H):
    """
    Generate an initial guess by diagonalizing the core Hamiltonian.

    Parameters
    ----------
    system : forte2.System
        The system object containing the atoms and basis set.
    H : NDArray
        The core Hamiltonian matrix.

    Returns
    -------
    NDArray
        The initial MO guess for the SCF procedure.
    """
    Xorth = system.get_Xorth()
    Htilde = Xorth.T @ H @ Xorth
    _, C = np.linalg.eigh(Htilde)
    return Xorth @ C


def guess_mix(C, homo_idx, mixing_parameter=np.pi / 4):
    """
    Induce the breaking of S^2 symmetry for UHF ms=0.0 calculations.
    This is helpful for obtaining proxies for open-shell singlets, for example.

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    homo_idx : int
        The index of the highest occupied molecular orbital (HOMO).
    mixing_parameter : float, optional
        The mixing parameter for the Givens rotation.
    twocomp : bool, optional
        Whether the system is two-component.

    Returns
    -------
    NDArray
        The modified MO coefficients.

    Notes
    -----
    See Szabo and Ostlund Ch. 3.8.7.

    """
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    Ca = givens_rotation(C, cosq, sinq, homo_idx, homo_idx + 1)
    Cb = givens_rotation(C, cosq, -sinq, homo_idx, homo_idx + 1)
    return [Ca, Cb]

def guess_mix_ghf(C, ha, hb, la, lb, mixing_parameter=np.pi / 4):
    """
    Induce the breaking of S^2 symmetry for UHF ms=0.0 calculations.
    This is helpful for obtaining proxies for open-shell singlets, for example.

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    ha : int
        The index of the highest occupied alpha(-like) orbital.
    hb : int
        The index of the highest occupied beta(-like) orbital.
    la : int
        The index of the lowest unoccupied alpha(-like) orbital.
    lb : int
        The index of the lowest unoccupied beta(-like) orbital.
    mixing_parameter : float, optional
        The mixing parameter for the Givens rotation.

    Returns
    -------
    NDArray
        The modified MO coefficients.

    Notes
    -----
    See Szabo and Ostlund Ch. 3.8.7.

    """
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    # alpha channel
    C = givens_rotation(C, cosq, sinq, ha, la)
    # beta channel
    C = givens_rotation(C, cosq, -sinq, hb, lb)
    return C


def alpha_beta_mix(C, mixing_parameter=0.1):
    """
    Induce the breaking of S_z symmetry for GHF calculations.
    This function explicitly mixes the degenerate alpha and beta MOs,
    which results in non-vanishing D_ab/D_ba density matrix elements.

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    mixing_parameter : float, optional
        The mixing parameter for the Givens rotation.

    Returns
    -------
    NDArray
        The modified MO coefficients.
    """
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    nmo = C.shape[1]
    for i in range(0, nmo, 2):
        C = givens_rotation(C, cosq, sinq, i, i + 1)
    return C


def break_complex_conjugation_symmetry(C, pert_strength=0.1):
    """
    Break the time-reversal/complex conjugation symmetry of the MO coefficients.
    A random phase is applied to each AO (cannot be MO as it would not change the density matrix).

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    pert_strength : float, optional
        The strength of the perturbation (in radians).

    Returns
    -------
    NDArray
        The modified MO coefficients.
    """
    # make sure alpha and beta are not complex conjugates
    phi = np.random.uniform(low=-pert_strength, high=pert_strength, size=C.shape[1])
    U = np.diag(np.exp(1.0j * phi))
    C = U @ C
    return C


def convert_coeff_spatial_to_spinor(system, C, complex=True):
    """
    Convert spatial orbital MO coefficients to spinor(bital) MO coefficients

    Parameters
    ----------
    system : forte2.System
        The system object containing the atoms and basis set.
    C : list of NDArray
        The MO coefficients in spatial orbital basis.
    complex : bool, optional, default=True
        Whether to cast to complex dtype.

    Returns
    -------
    list of NDArray
        The MO coefficients in spinor(bital) basis.
    """
    dtype = np.complex128 if complex else np.float64
    nbf = system.nbf
    C_2c = np.zeros((nbf * 2,) * 2, dtype=dtype)
    assert isinstance(C, list)
    if len(C) == 2:
        # UHF
        assert C[0].shape[0] == nbf
        assert C[1].shape[0] == nbf
        # |a^0_{alfa AO} b^0_{alfa AO} ... |
        # |a^0_{beta AO} b^0_{beta AO} ... |
        C_2c[:nbf, ::2] = C[0]
        C_2c[nbf:, 1::2] = C[1]
    elif len(C) == 1:
        # RHF/ROHF
        C_2c[:nbf, ::2] = C[0]
        C_2c[nbf:, 1::2] = C[0]
    else:
        raise RuntimeError(f"Coefficient of length {len(C)} not recognized!")
    return [C_2c]
