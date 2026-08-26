import numpy as np
from numpy.typing import NDArray
import scipy as sp

import forte2.integrals as integrals
from forte2.helpers.matrix_functions import compute_Am1y
from forte2.lib import ints


def flat_to_atom_gradient(gradient, natoms):
    """
    Convert a flat atom-major Cartesian gradient to ``(natoms, 3)`` shape.

    Parameters
    ----------
    gradient : array_like
        Flat gradient vector with shape ``(3 * natoms,)``.
    natoms : int
        Number of atoms.

    Returns
    -------
    NDArray
        Gradient array with shape ``(natoms, 3)``.
    """
    gradient = np.asarray(gradient, dtype=float)
    expected_shape = (3 * natoms,)
    if gradient.shape != expected_shape:
        raise ValueError(
            f"Expected a flat gradient of shape {expected_shape}, got {gradient.shape}."
        )
    return gradient.reshape(natoms, 3).copy()


def nuclear_repulsion_deriv(atoms):
    r"""
    Compute point-charge nuclear repulsion derivatives.

    The derivative is returned in Hartree/Bohr for coordinates in Bohr:

    .. math::
        \frac{\partial E_\mathrm{nuc}}{\partial R_{A\alpha}}
        =
        -\sum_{B \ne A} Z_A Z_B
        \frac{R_{A\alpha} - R_{B\alpha}}{|\mathbf{R}_A-\mathbf{R}_B|^3}.

    Parameters
    ----------
    atoms : list[tuple[float, Sequence[float]]]
        Nuclear charges and Cartesian centers.

    Returns
    -------
    NDArray
        Nuclear repulsion derivative with shape ``(natoms, 3)``.
    """
    natoms = len(atoms)
    charges = np.asarray([atom[0] for atom in atoms], dtype=float)
    positions = np.asarray([atom[1] for atom in atoms], dtype=float)
    gradient = np.zeros((natoms, 3), dtype=float)

    for a in range(natoms):
        for b in range(a + 1, natoms):
            rab = positions[a] - positions[b]
            distance = np.linalg.norm(rab)
            if distance < 1.0e-14:
                raise ValueError(
                    "Nuclear repulsion derivative is undefined for coincident nuclei."
                )
            contribution = charges[a] * charges[b] * rab / distance**3
            gradient[a] -= contribution
            gradient[b] += contribution

    return gradient


def _system_nuclear_repulsion_deriv(system):
    """Compute nuclear repulsion derivatives for the system's nuclear model."""
    if not system.use_gaussian_charges:
        return nuclear_repulsion_deriv(system.atoms)

    weights = 0.5 * np.outer(system.atomic_charges, system.atomic_charges)
    np.fill_diagonal(weights, 0.0)
    return flat_to_atom_gradient(
        ints.coulomb_2c_deriv(
            system.gaussian_charge_basis,
            system.gaussian_charge_basis,
            weights,
            system.atoms,
        ),
        system.natoms,
    )


def compute_gradient(
    system,
    D1,
    W1,
    W2,
    W3,
    hcore_gradient=None,
):
    r"""
    Compute the total gradient from the one-electron density matrix and two-electron derivative weights.

    The returned gradient is in Hartree/Bohr for coordinates in Bohr.

    Parameters
    ----------
    system : System
        The system for which to compute the gradient.
    D1 : NDArray
        The one-electron density matrix with shape ``(nbasis, nbasis)``.
    W1 : NDArray
        The energy-weighted density matrix with shape ``(nbasis, nbasis)``.
    W2 : NDArray
        The two-electron derivative weight for the metric with shape ``(naux, naux)``.
    W3 : NDArray
        The two-electron derivative weight for the three-center integrals with shape ``(naux, nbasis, nbasis)``.
    hcore_gradient : NDArray, optional
        Precontracted one-electron Hamiltonian contribution with shape
        ``(natoms, 3)``. When provided, this replaces the separate kinetic-
        and nuclear-attraction derivative contractions.

    Returns
    -------
    NDArray
        Total gradient with shape ``(natoms, 3)``.
    """
    natoms = system.natoms
    gradient = _system_nuclear_repulsion_deriv(system)
    if hcore_gradient is not None:
        hcore_gradient = np.asarray(hcore_gradient, dtype=float)
        if hcore_gradient.shape != (natoms, 3):
            raise ValueError(
                f"Expected hcore_gradient shape {(natoms, 3)}, "
                f"got {hcore_gradient.shape}."
            )
        gradient += hcore_gradient
    else:
        gradient += flat_to_atom_gradient(
            ints.kinetic_deriv(system.basis, system.basis, D1, system.atoms),
            natoms,
        )
        gradient += flat_to_atom_gradient(integrals.nuclear_deriv(system, D1), natoms)
    gradient -= flat_to_atom_gradient(
        ints.overlap_deriv(system.basis, system.basis, W1, system.atoms), natoms
    )
    # Build the two-electron derivative weights and contract with the integrals.
    gradient += flat_to_atom_gradient(integrals.coulomb_3c_deriv(system, W3), natoms)
    gradient += flat_to_atom_gradient(integrals.coulomb_2c_deriv(system, W2), natoms)
    return gradient
