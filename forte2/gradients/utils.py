import numpy as np


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
    ndarray
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
    ndarray
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
