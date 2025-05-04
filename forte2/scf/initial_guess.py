import numpy as np
import scipy as sp

from forte2.system.build_basis import assemble_basis
import forte2


def minao_initial_guess(system, H, S):
    """
    Generate a minao initial guess for the density matrix.
    """
    # Get the number of basis functions
    nbasis = system.basis.size

    # Initialize the density matrix
    D = np.zeros((nbasis, nbasis))

    sap_basis = assemble_basis(
        "sap_helfem_large", system.atoms, embed_normalization_into_coefficients=False
    )

    for shell in sap_basis:
        print(shell)

    # scale the sap basis
    scaled_sap_basis = forte2.ints.Basis()

    for shell in sap_basis:
        # scale the coefficients

        scaled_coeff = np.array(
            [-c * ((e / np.pi) ** 1.5) for c, e in zip(shell.coeff, shell.exponents)]
        )

        scaled_shell = forte2.ints.Shell(
            shell.l,
            shell.exponents,
            scaled_coeff,
            shell.center,
            shell.is_pure,
            embed_normalization_into_coefficients=False,
        )

        scaled_sap_basis.add(scaled_shell)

    for shell in scaled_sap_basis:
        print(shell)

    SAP_ints = forte2.ints.coulomb_3c(scaled_sap_basis, system.basis, system.basis)
    SAP_V = np.einsum("Pmn->mn", SAP_ints)

    H_SAP = H + SAP_V

    # Diagonalize the Fock matrix
    eps, C = sp.linalg.eigh(H_SAP, S)

    return C
