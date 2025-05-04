import numpy as np
from numpy.typing import NDArray
import scipy as sp

from forte2.system.build_basis import assemble_basis
import forte2


def minao_initial_guess(system: forte2.System, H: NDArray, S: NDArray) -> NDArray:
    """
    Generate a superposition of atomic potentials (SAP) initial guess for the SCF procedure
    S. Lehtola, J. Chem. Theory Comput. 15, 1593-1604 (2019), arXiv:1810.11659.
    For details, see https://doi.org/10.1063/5.0004046

    Parameters
    ----------
    system : forte2.System
        The system object containing the atoms and basis set.
    H : np.ndarray
        The Fock matrix.
    S : np.ndarray
        The overlap matrix.
    Returns
    -------
    np.ndarray
        The initial MO guess for the SCF procedure.
    """

    # generate the SAP basis from the initial guess file. Skip normalization
    sap_basis = assemble_basis(
        "sap_helfem_large", system.atoms, embed_normalization_into_coefficients=False
    )

    # create a new basis that will be used to store the scaled coefficients
    scaled_sap_basis = forte2.ints.Basis()

    for shell in sap_basis:
        # scales the coefficients by -(exponent / pi)^(3/2)
        scaled_coeff = np.array(
            [-c * ((e / np.pi) ** 1.5) for c, e in zip(shell.coeff, shell.exponents)]
        )
        scaled_shell = forte2.ints.Shell(
            shell.l,
            shell.exponents,
            scaled_coeff,
            shell.center,
            shell.is_pure,
            embed_normalization_into_coefficients=False,  # do not normalize
        )
        scaled_sap_basis.add(scaled_shell)

    # generate the SAP integrals (P|mn)
    SAP_ints = forte2.ints.coulomb_3c(scaled_sap_basis, system.basis, system.basis)

    # generate the SAP potential V_mn = sum_P (P|mn)
    SAP_V = np.einsum("Pmn->mn", SAP_ints)

    # generate the SAP Hamiltonian and diagonalize it
    H_SAP = H + SAP_V
    eps, C = sp.linalg.eigh(H_SAP, S)

    return C
