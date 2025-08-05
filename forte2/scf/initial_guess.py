import numpy as np
import scipy as sp

from forte2 import ints, Basis, Shell
from forte2.system import System
from forte2.system.build_basis import build_basis


def minao_initial_guess(system, H, S):
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
    sap_basis = build_basis(
        "sap_helfem_large", system.atoms, embed_normalization_into_coefficients=False
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


def core_initial_guess(system: System, H, S):
    Htilde = system.Xorth.T @ H @ system.Xorth
    _, C = np.linalg.eigh(Htilde)
    return system.Xorth @ C
