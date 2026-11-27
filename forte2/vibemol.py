"""Prepare Forte2 calculation results for the VibeMol notebook viewer."""

from dataclasses import dataclass

import numpy as np

from forte2.mcopt import MCOptimizer
from forte2.orbitals import molden_string
from forte2.scf import GHF, RHF, UHF
from forte2.scf.scf_base import SCFBase


def orbitals_molden(method, *, title=None) -> tuple[str, str]:
    """Export geometry and molecular orbitals as a named Molden document.

    Parameters
    ----------
    method : Method
        Executed method supported by the Forte2 Molden writer.
    title : str, optional
        Molden title and filename stem. The default filename stem is the
        lowercase method class name.

    Returns
    -------
    tuple[str, str]
        Filename ending in ``.molden`` and the Molden text.
    """
    name = title or type(method).__name__.lower()
    return f"{name}.molden", molden_string(method, title=title)


def atomic_numbers(method) -> list[int]:
    """Return the atomic numbers in the method's System atom order."""
    return [int(number) for number, _ in method.system.atoms]


def forces(method, root=None) -> np.ndarray:
    """Return the negative analytic nuclear gradient in System atom order.

    Parameters
    ----------
    method : MCOptimizer, RHF, UHF or GHF
        Method supplying the analytic nuclear gradient.
    root : int, optional
        Zero-based absolute root index. Required for a multiroot MCOptimizer.
        Use None for SCF.

    Returns
    -------
    numpy.ndarray
        Cartesian forces with shape ``(natoms, 3)`` in Hartree per bohr.

    Raises
    ------
    TypeError
        If the method does not support this adapter.
    ValueError
        If the root is invalid for the method or the gradient is not finite,
        real, and shaped like the System coordinates.
    """
    if isinstance(method, MCOptimizer):
        gradient = method.gradient(root=root)
    elif isinstance(method, (RHF, UHF, GHF)):
        if root is not None:
            raise ValueError("root must be None for SCF forces.")
        gradient = method.gradient()
    else:
        raise TypeError("Forces require MCOptimizer, RHF, UHF, or GHF.")

    gradient = np.asarray(gradient)
    if np.iscomplexobj(gradient) or not np.all(np.isfinite(gradient)):
        raise ValueError("The gradient must be real and finite.")
    shape = (len(method.system.atoms), 3)
    if gradient.shape != shape:
        raise ValueError(
            f"Expected a gradient with shape {shape}, got {gradient.shape}."
        )
    return -gradient


def _transition_solver(method):
    solver = getattr(method, "ci_solver", None)
    if (
        not getattr(method, "executed", False)
        or not getattr(method, "do_transition_dipole", False)
        or getattr(solver, "transition_dipoles", None) is None
    ):
        raise ValueError("Run the active-space driver with do_transition_dipole=True.")
    return solver


def state_dipole(method, root) -> np.ndarray:
    """Return the total electric dipole of a neutral state in atomic units.

    Parameters
    ----------
    method : ActiveSpaceDriver
        Executed driver with ``do_transition_dipole=True`` and a neutral SCF
        reference.
    root : int
        Zero-based absolute root index.

    Returns
    -------
    numpy.ndarray
        Electronic plus nuclear dipole in units of electron charge times bohr.

    Raises
    ------
    ValueError
        If the dipole is unavailable, the system is charged, or the reference
        charge cannot be determined.
    """
    solver = _transition_solver(method)
    reference = method
    while reference is not None and not isinstance(reference, SCFBase):
        reference = getattr(reference, "parent_method", None)
    charge = getattr(reference, "charge", None)
    if charge is None:
        raise ValueError("Cannot determine the SCF reference charge.")
    if charge != 0:
        raise ValueError("The state dipole of a charged system depends on the origin.")

    dipoles = solver.transition_dipoles
    key = (root, root)
    if key not in dipoles:
        available = sorted(i for i, j in dipoles if i == j)
        raise ValueError(
            f"No state dipole for root {root}. Available roots: {available}."
        )
    return np.array(dipoles[key])


def transition_dipole(method, i, j) -> tuple[np.ndarray, float]:
    """Return a transition dipole and oscillator strength.

    Parameters
    ----------
    method : ActiveSpaceDriver
        Executed driver with ``do_transition_dipole=True``.
    i, j : int
        Zero-based absolute root indices, in either order.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Dipole in atomic units and dimensionless oscillator strength for the
        stored lower- to higher-energy transition.

    Raises
    ------
    ValueError
        If the driver did not compute transition properties or the pair is
        unavailable.
    """
    solver = _transition_solver(method)
    dipoles = solver.transition_dipoles
    strengths = solver.oscillator_strengths
    available = dipoles.keys() & strengths.keys()
    for key in ((i, j), (j, i)):
        if key in available:
            return np.array(dipoles[key]), float(strengths[key])
    raise ValueError(
        f"No transition for {(i, j)}. Available pairs: {sorted(available)}."
    )


@dataclass(frozen=True)
class Shown:
    """Handle to the last queued command and its applied vector scale.

    Attributes
    ----------
    future : object
        Future returned by the viewer's last command.
    scale : float or None
        Applied scale in angstrom per atomic unit of the quantity shown.
        None when the command did not specify a scale.
    """

    future: object
    scale: float | None = None


def _require_sequential_commands(viewer):
    if getattr(viewer, "sequential_commands", False) is not True:
        raise RuntimeError(
            "forte2.vibemol needs vibemol-widget 0.2.0 or later, "
            "which runs commands in order"
        )


def _scale_for_length(magnitude, length, quantity):
    if magnitude == 0:
        raise ValueError(f"Cannot scale zero {quantity}. Supply an explicit scale.")
    return length / float(magnitude)


def show_orbitals(viewer, method, *, indices=None, title=None) -> Shown:
    """Load a method's Molden document and optionally show selected orbitals.

    Parameters
    ----------
    viewer : object
        Notebook viewer supporting sequential commands.
    method : Method
        Executed method supported by the Molden writer.
    indices : iterable of int, optional
        One-based Molden indices to show in the current viewer mode.
    title : str, optional
        Molden title and filename stem.

    Returns
    -------
    Shown
        Future for loading, or for showing the selected orbitals if given.
    """
    _require_sequential_commands(viewer)
    document = orbitals_molden(method, title=title)
    future = viewer.load(document, clear_first=True)
    if indices is not None:
        future = viewer.show_orbitals(indices)
    return Shown(future)


def show_forces(
    viewer,
    method,
    root=None,
    *,
    scale=None,
    max_length=1.0,
    name="forces",
    color="#B22D73",
) -> Shown:
    """Draw one force arrow per atom on the viewer's active structure.

    Parameters
    ----------
    viewer : object
        Notebook viewer supporting sequential commands.
    method : MCOptimizer, RHF, UHF or GHF
        Method supplying forces in the active structure's System atom order.
    root : int, optional
        Zero-based root for MCOptimizer. Use None for SCF.
    scale : float, optional
        Angstrom per Hartree/bohr. If omitted, scale the largest force to
        ``max_length``.
    max_length : float, optional
        Length in angstrom of the largest force arrow for automatic scaling.
    name, color : str, optional
        Vector set name and color.

    Returns
    -------
    Shown
        Future and applied scale, which can be reused for another state.
    """
    _require_sequential_commands(viewer)
    values = forces(method, root=root)
    if scale is None:
        scale = _scale_for_length(
            np.linalg.norm(values, axis=1).max(), max_length, "forces"
        )
    future = viewer.set_atom_vectors(
        name, values.tolist(), color, scale=scale, atomic_numbers=atomic_numbers(method)
    )
    return Shown(future, scale)


def show_dipole(
    viewer, method, root, *, scale=None, length=1.5, name="dipole", color="#2365C6"
) -> Shown:
    """Draw a neutral state's total dipole at the center of mass.

    Parameters
    ----------
    viewer : object
        Notebook viewer supporting sequential commands.
    method : ActiveSpaceDriver
        Executed driver with ``do_transition_dipole=True``.
    root : int
        Zero-based absolute root index.
    scale : float, optional
        Angstrom per atomic unit of dipole. If omitted, use ``length``
        to scale the dipole.
    length : float, optional
        Arrow length in angstrom for automatic scaling.
    name, color : str, optional
        Arrow name and color.

    Returns
    -------
    Shown
        Future and applied scale.

    Raises
    ------
    ValueError
        If the dipole is unavailable or zero without an explicit scale.
    """
    _require_sequential_commands(viewer)
    vector = state_dipole(method, root)
    if scale is None:
        scale = _scale_for_length(np.linalg.norm(vector), length, "dipole")
    future = viewer.set_arrow(name, vector.tolist(), color, scale=scale)
    return Shown(future, scale)


def show_transition_dipole(
    viewer, method, i, j, *, length=2.4, name="transition dipole", color="#208375"
) -> Shown:
    """Draw a two-headed transition dipole whose overall sign is arbitrary.

    Parameters
    ----------
    viewer : object
        Notebook viewer supporting sequential commands.
    method : ActiveSpaceDriver
        Executed driver with ``do_transition_dipole=True``.
    i, j : int
        Zero-based absolute root indices, in either order.
    length : float, optional
        Arrow length in angstrom.
    name, color : str, optional
        Arrow name and color.

    Returns
    -------
    Shown
        Future for creating the arrow.
    """
    _require_sequential_commands(viewer)
    vector, _ = transition_dipole(method, i, j)
    future = viewer.set_arrow(name, vector.tolist(), color, length=length, heads="both")
    return Shown(future)
