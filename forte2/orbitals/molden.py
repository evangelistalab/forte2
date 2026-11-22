"""Molden-format writer for molecular orbitals.

Conventions
-----------
Molden files are read by many programs (Molden, Jmol, VibeMol, PySCF, IboView,
Multiwfn, ...). This writer follows the conventions those readers share; the
test suite checks the written orbitals point by point against
:func:`forte2.lib.ints.orbitals_on_grid`.

* ``[Atoms] AU``: coordinates in bohr.
* ``[GTO]``: contraction coefficients multiply *normalized* primitive
  Gaussians. Forte2 shells store coefficients with the primitive normalization
  folded in, so it is divided out here. The contracted functions written this
  way are normalized, so readers that renormalize contractions and readers that
  do not reconstruct the same orbitals.
* Pure (spherical) shells only. ``p`` functions are written as ``(x, y, z)``;
  shells with ``l >= 2`` use the Molden order ``m = 0, +1, -1, +2, -2, ...``.
  Forte2 stores pure shells in the Libint/CCA order ``m = -l, ..., +l``, in
  which ``p`` is ``(y, z, x)``.

Orbital energies and occupations
--------------------------------
Every orbital needs an energy and an occupation. Where they come from depends
on the method that produced the orbitals:

* SCF references: the orbital energies, and occupations projected from the
  SCF density matrices.
* Active-space drivers and solvers (``CI``, ``MCOptimizer``, ``CISolver``,
  ...): the generalized-Fock diagonal; 2 for core orbitals, the diagonal of
  the state-averaged 1-RDM for active orbitals, 0 for virtual orbitals.
* Orbital rotations of an SCF reference (``AVAS``): the reference Fock
  diagonal and occupations projected from the reference density.

Methods downstream of a correlated calculation (``ASET``, ``DSRG_MRPT2``) are
rejected rather than given SCF-derived values. Their orbitals can still be
written by passing the ``System`` and explicit arrays.
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forte2.base_classes import ActiveSpaceDriver, ActiveSpaceSolver, Method
from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.integrals import overlap
from forte2.scf.scf_base import SCFBase
from forte2.system import System
from forte2.system.basis_utils import AM_LABELS, ml_from_shell_index_cca

__all__ = ["write_molden", "molden_string"]

MAX_MOLDEN_L = 4
"""The Molden format defines shells up to g."""

_SPIN_LABELS = ("Alpha", "Beta")
_CORRELATED = (ActiveSpaceDriver, ActiveSpaceSolver)
_IMAG_TOL = 1e-10
"""Largest imaginary part of a coefficient that is treated as numerical noise."""
_OCC_ZERO_TOL = 1e-10
"""Projected occupations smaller than this are written as exactly zero."""
_CENTER_TOL = 1e-6
"""Largest distance (bohr) between a basis-function center and its atom."""


@dataclass(frozen=True)
class _MoldenBlock:
    """One spin block of MOs, with coefficients in Forte2's AO order."""

    C: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    symmetries: tuple[str, ...]
    spin: str


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def write_molden(source, path="orbitals.molden", **kwargs) -> Path:
    """
    Write molecular orbitals to a Molden file.

    Parameters
    ----------
    source : Method or System
        An executed Forte2 method that provides ``system`` and ``mos`` (SCF
        references, AVAS, and active-space drivers and solvers such as ``CI``,
        ``MCOptimizer`` and ``CISolver``), or a ``System`` together with
        ``C=...``.
    path : str or pathlib.Path, optional, default="orbitals.molden"
        Destination path.
    **kwargs
        Passed to :func:`molden_string` (``C``, ``energies``, ``occupations``,
        ``symmetries``, ``title``, ``split_spin_blocks``).

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    path = Path(path)
    path.write_text(molden_string(source, **kwargs), encoding="ascii")
    return path


def molden_string(
    source,
    *,
    C=None,
    energies=None,
    occupations=None,
    symmetries=None,
    title=None,
    split_spin_blocks=False,
) -> str:
    """
    Return molecular orbitals as Molden-format text.

    Parameters
    ----------
    source : Method or System
        An executed Forte2 method, or a ``System``. See :func:`write_molden`.
    C : ndarray or list of ndarray, optional
        Required with a ``System`` and not allowed with a method. Finite real
        AO-basis coefficients, shape ``(nbf, nmo)`` with at least one orbital.
        Pass one array for restricted orbitals or ``[C_alpha, C_beta]`` for
        unrestricted ones.
    energies, occupations : ndarray or list of ndarray, optional
        Finite real per-orbital values with the same block structure as ``C``.
        Default to zero.
    symmetries : list of str or list of list of str, optional
        Per-orbital symmetry labels. Default to ``"A"``.
    title : str, optional
        Text of the ``[Title]`` section.
    split_spin_blocks : bool, optional, default=False
        Write a separate ``[MO]`` section for each spin block. PySCF needs this
        layout when the total number of alpha and beta orbitals equals the
        number of AOs. The default single section follows the Molden format.

    Returns
    -------
    str
        The Molden file contents.

    Raises
    ------
    TypeError
        If ``source`` is neither a method nor a system, or if orbital arrays
        are combined with a method.
    ValueError
        If coefficients or per-orbital data are empty, nonfinite, or have an
        invalid shape.
    RuntimeError
        If the method has not been run or does not provide orbitals.
    NotImplementedError
        For two-component or complex orbitals, orbitals derived from a
        correlated calculation by a method other than its driver or solver,
        Cartesian or generally contracted shells, or shells beyond g.
    """
    system, blocks, default_title = _collect_blocks(
        source, C, energies, occupations, symmetries
    )
    basis = system.basis
    _check_basis(basis)
    lines = ["[Molden Format]", "[Title]", _ascii_line(title or default_title), ""]
    lines += _atoms_section(system)
    lines += _pure_shell_tags(basis)
    lines += _gto_section(system)
    lines += _mo_sections(blocks, _molden_ao_order(basis), split_spin_blocks)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orbital blocks
# --------------------------------------------------------------------------- #


def _collect_blocks(source, C, energies, occupations, symmetries):
    """Return ``(system, blocks, default_title)`` for a method or a system."""
    arrays_given = any(x is not None for x in (C, energies, occupations, symmetries))
    if isinstance(source, Method):
        if arrays_given:
            raise TypeError(
                "C, energies, occupations and symmetries can only be combined with a "
                "System. To write other orbitals of a method, pass method.system."
            )
        blocks = _method_blocks(source)  # validates the method first
        return source.system, blocks, f"Forte2 {type(source).__name__} orbitals"
    if isinstance(source, System):
        if C is None:
            raise TypeError("C is required when writing the orbitals of a System.")
        return (
            source,
            _array_blocks(source, C, energies, occupations, symmetries),
            "Forte2 orbitals",
        )
    raise TypeError(
        f"molden_string() expects a forte2 Method or System, got {type(source).__name__}."
    )


def _method_blocks(method) -> list[_MoldenBlock]:
    """Orbitals of an executed method, with energies and occupations."""
    _check_method(method)
    system = method.system
    C_blocks = [_coefficients(C, system, full_space=True) for C in method.mos.C]
    nmo = C_blocks[0].shape[1]
    symmetries = [_irrep_labels(labels, nmo) for labels in method.mos.irrep_labels]

    if isinstance(method, _CORRELATED):
        _require_restricted(method, C_blocks)
        energies, occupations = _active_space_data(method, C_blocks[0])
        return _spin_blocks(C_blocks, [energies], [occupations], symmetries)

    if isinstance(method, SCFBase):
        return _spin_blocks(
            C_blocks,
            _scf_energies(method, nmo),
            _projected_occupations(method, system, C_blocks),
            symmetries,
        )

    if "mo_space" in method.provides:
        # An orbital rotation of an SCF reference, such as AVAS.
        _require_restricted(method, C_blocks)
        scf = _upstream_scf(method)
        return _spin_blocks(
            C_blocks,
            [_fock_diagonal(method, scf, C_blocks[0])],
            _projected_occupations(scf, system, C_blocks),
            symmetries,
        )

    raise NotImplementedError(
        f"write_molden() does not know the occupations of {type(method).__name__} "
        "orbitals. Pass method.system with C=..., energies=..., occupations=... instead."
    )


def _array_blocks(system, C, energies, occupations, symmetries) -> list[_MoldenBlock]:
    """Orbitals given as arrays, for a ``System``."""
    C_blocks = [C] if isinstance(C, np.ndarray) and C.ndim == 2 else list(C)
    if len(C_blocks) not in (1, 2):
        raise ValueError("C must be one coefficient matrix or [C_alpha, C_beta].")
    C_blocks = [_coefficients(c, system, full_space=False) for c in C_blocks]
    nblocks, nmo = len(C_blocks), C_blocks[0].shape[1]
    if any(c.shape[1] != nmo for c in C_blocks):
        raise ValueError(
            "Alpha and beta coefficient matrices must have the same shape."
        )
    return _spin_blocks(
        C_blocks,
        _value_blocks(energies, "energies", nblocks, nmo),
        _value_blocks(occupations, "occupations", nblocks, nmo),
        _symmetry_blocks(symmetries, nblocks, nmo),
    )


def _spin_blocks(C_blocks, energies, occupations, symmetries) -> list[_MoldenBlock]:
    """Pair each coefficient block with its values and spin label."""
    return [
        _MoldenBlock(C, e, n, tuple(str(s) for s in sym), spin)
        for C, e, n, sym, spin in zip(
            C_blocks,
            energies,
            occupations,
            symmetries,
            _SPIN_LABELS[: len(C_blocks)],
            strict=True,
        )
    ]


# --------------------------------------------------------------------------- #
# Validation and input normalization
# --------------------------------------------------------------------------- #


def _check_method(method):
    name = type(method).__name__
    missing = {"system", "mos"} - method.provides
    if missing:
        raise RuntimeError(
            f"Method {name} does not provide required data: {', '.join(sorted(missing))}."
        )
    if not method.executed:
        raise RuntimeError(
            f"{name} has not been run. Call run() before writing orbitals."
        )
    # Use the method's own flag: System.two_component is mutated in place when
    # a shared System is bound to another (e.g. GHF) method.
    if method.two_component or method.mos.spinorbital:
        raise NotImplementedError(
            "Two-component orbitals cannot be written in Molden format. Use "
            "forte2.write_orbital_cubes(..., format=('2ccube',)) instead."
        )


def _require_restricted(method, C_blocks):
    if len(C_blocks) != 1:
        raise NotImplementedError(
            f"write_molden() supports {type(method).__name__} only with restricted "
            "orbitals."
        )


def _coefficients(C, system, full_space) -> np.ndarray:
    """Validate an AO-basis coefficient matrix and return it as a real array."""
    C = np.asarray(C)
    if C.ndim != 2 or C.shape[0] != system.nbf:
        hint = (
            " Two-component orbitals cannot be written in Molden format."
            if C.ndim == 2 and C.shape[0] == 2 * system.nbf
            else ""
        )
        raise ValueError(
            f"MO coefficients must have shape (nbf={system.nbf}, nmo), got {C.shape}.{hint}"
        )
    if C.shape[1] == 0:
        raise ValueError("MO coefficients must contain at least one orbital.")
    if full_space and C.shape[1] != system.nmo:
        raise RuntimeError("The MO coefficient matrix must span the full MO space.")
    C = np.asarray(C, dtype=complex if np.iscomplexobj(C) else float)
    if not np.isfinite(C).all():
        raise ValueError("MO coefficients must contain only finite values.")
    if np.iscomplexobj(C) and np.abs(C.imag).max() > _IMAG_TOL:
        raise NotImplementedError(
            "Complex-valued orbitals cannot be written in Molden format."
        )
    return C.real


def _irrep_labels(labels, nmo) -> list[str]:
    """A method's irrep labels, or ``"A"`` for each orbital if they don't fit."""
    labels = [str(x) for x in labels] if labels is not None else []
    return labels if len(labels) == nmo else ["A"] * nmo


def _value_blocks(values, name, nblocks, nmo) -> list[np.ndarray]:
    """Per-orbital values for each block; zeros when not given."""
    if values is None:
        return [np.zeros(nmo) for _ in range(nblocks)]
    blocks = [values] if not len(values) or np.ndim(values[0]) == 0 else list(values)
    if len(blocks) != nblocks:
        raise ValueError(f"{name} must have one block per coefficient matrix.")
    out = []
    for block in blocks:
        block = np.asarray(block)
        if block.shape != (nmo,):
            raise ValueError(
                f"Expected {nmo} {name} per block, got shape {block.shape}."
            )
        if np.iscomplexobj(block):
            raise ValueError(f"{name} must contain only real values.")
        block = block.astype(float, copy=False)
        if not np.isfinite(block).all():
            raise ValueError(f"{name} must contain only finite values.")
        out.append(block)
    return out


def _symmetry_blocks(symmetries, nblocks, nmo) -> list[list[str]]:
    """Per-orbital symmetry labels for each block; ``"A"`` when not given."""
    if symmetries is None:
        return [["A"] * nmo for _ in range(nblocks)]
    blocks = (
        [symmetries]
        if len(symmetries) and isinstance(symmetries[0], str)
        else symmetries
    )
    blocks = [list(block) for block in blocks]
    if len(blocks) != nblocks or any(len(block) != nmo for block in blocks):
        raise ValueError("symmetries must provide one label per orbital and block.")
    return blocks


# --------------------------------------------------------------------------- #
# Energies and occupations
# --------------------------------------------------------------------------- #


def _scf_energies(scf, nmo) -> list[np.ndarray]:
    energies = [
        np.asarray(np.real_if_close(np.asarray(e)), dtype=float) for e in scf.eps
    ]
    if any(e.shape != (nmo,) for e in energies):
        raise RuntimeError("Orbital energies have invalid shapes.")
    return energies


def _upstream_scf(method) -> SCFBase:
    """The SCF reference that ``method`` rotates, found by walking up the chain.

    Raises if a correlated calculation sits in between: the orbitals then
    derive from it, and SCF-based occupations would be wrong.
    """
    name = type(method).__name__
    stage = getattr(method, "parent_method", None)
    while stage is not None:
        if isinstance(stage, SCFBase):
            return stage
        if isinstance(stage, _CORRELATED):
            raise NotImplementedError(
                f"The orbitals of {name} derive from a correlated "
                f"{type(stage).__name__} calculation, so write_molden() cannot "
                "assign them energies and occupations. Write the orbitals of that "
                "calculation instead (for example write_molden(mc)), or pass "
                "method.system with C=..., energies=..., occupations=...."
            )
        stage = getattr(stage, "parent_method", None)
    raise NotImplementedError(
        f"Could not find the SCF reference of {name} to define orbital "
        "occupations. Pass method.system with C=... and occupations=... instead."
    )


def _projected_occupations(scf, system, C_blocks) -> list[np.ndarray]:
    """Occupations n_i = c_i^T S D S c_i in the SCF spin densities.

    A single block of spatial orbitals gets alpha plus beta; two blocks get
    one spin density each.
    """
    S = np.asarray(overlap(system))
    Da, Db = _spin_densities(scf)
    if len(C_blocks) == 1:
        (C,) = C_blocks
        return [np.clip(_project(C, S, Da) + _project(C, S, Db), 0.0, 2.0)]
    Ca, Cb = C_blocks
    return [_project(Ca, S, Da), _project(Cb, S, Db)]


def _spin_densities(scf):
    """``[D_alpha, D_beta]`` in the AO basis (RHF stores only ``D_alpha``)."""
    D = [np.asarray(d) for d in scf.D]
    return D * 2 if len(D) == 1 else D


def _project(C, S, D) -> np.ndarray:
    SC = S @ C
    occ = np.einsum("mi,mn,ni->i", SC, D, SC).real
    occ[np.abs(occ) < _OCC_ZERO_TOL] = 0.0
    return occ


def _fock_diagonal(method, scf, C) -> np.ndarray:
    """Diagonal of the reference Fock matrix in the orbitals ``C``."""
    if len(scf.F) != 1:
        raise NotImplementedError(
            f"write_molden() supports {type(method).__name__} only on a restricted "
            "(RHF or ROHF) reference."
        )
    F = np.asarray(scf.F[0])
    return np.real_if_close(np.einsum("mi,mn,ni->i", C, F, C)).astype(float)


def _active_space_data(method, C):
    """Energies and occupations for active-space drivers and solvers.

    Core orbitals are doubly occupied, active orbitals carry the diagonal of the
    state-averaged one-particle density matrix (natural occupation numbers when
    ``final_orbitals="natural"``), virtual orbitals are empty. Energies are the
    diagonal of the generalized Fock matrix in the final orbital basis, which
    equals the semicanonical orbital energies when
    ``final_orbitals="semicanonical"``.
    """
    mo_space = method.mo_space
    g1 = np.asarray(method.make_average_rdm(1))
    if g1.shape != (mo_space.nactv, mo_space.nactv):
        raise RuntimeError("The active-space 1-RDM does not match the active space.")

    occupations = np.zeros(C.shape[1])
    occupations[np.asarray(mo_space.docc_indices, dtype=int)] = 2.0
    occupations[np.asarray(mo_space.active_indices, dtype=int)] = np.clip(
        np.real_if_close(np.diag(g1)).astype(float), 0.0, 2.0
    )
    return _generalized_fock_diagonal(method, C, g1), occupations


def _generalized_fock_diagonal(method, C, g1) -> np.ndarray:
    """Diagonal of the generalized Fock matrix in the orbitals ``C``.

    The AO matrix is built once from the core orbitals and the active 1-RDM and
    contracted with ``C``; no orbital blocks are diagonalized.
    """
    mo_space = method.mo_space
    fock_builder = _spatial_system(method.system).fock_builder
    F = fock_builder.build_generalized_fock(
        C_core=C[:, np.asarray(mo_space.docc_indices, dtype=int)],
        C_act=C[:, np.asarray(mo_space.active_indices, dtype=int)],
        g1=g1,
    )
    return np.real_if_close(np.einsum("mi,mn,ni->i", C, F, C)).astype(float)


def _spatial_system(system):
    """``system`` as seen by an already-run spatial (one-component) method.

    Binding a two-component method such as GHF sets ``two_component`` on a
    shared System in place, which changes ``ints_hcore()`` and the J/K builds of
    its Fock builder. In that case return a shallow copy with the flag cleared
    and its own shallow copy of the Fock builder: the shared System is not
    mutated, and the copies reuse the existing three-index integrals.
    """
    if not system.two_component:
        return system
    spatial = copy.copy(system)
    spatial.two_component = False
    spatial.fock_builder = copy.copy(system.fock_builder)
    spatial.fock_builder.system = spatial
    return spatial


# --------------------------------------------------------------------------- #
# Basis conventions
# --------------------------------------------------------------------------- #


def _check_basis(basis):
    for ishell in range(basis.nshells):
        shell = basis[ishell]
        if not shell.is_pure:
            raise NotImplementedError(
                "Cartesian shells are not supported in Molden output."
            )
        if shell.size != 2 * shell.l + 1:
            raise NotImplementedError(
                "Generally contracted shells are not supported in Molden output."
            )
        if shell.l > MAX_MOLDEN_L:
            raise NotImplementedError(
                f"The Molden format defines shells up to g; found l={shell.l}."
            )


def _primitive_norm(l, alpha) -> np.ndarray:
    """Normalization of the axial Cartesian primitive x^l exp(-alpha r^2).

    This is the factor Libint (and therefore Forte2) folds into shell
    coefficients, and the factor Molden readers apply to each primitive.
    """
    alpha = np.asarray(alpha, dtype=float)
    double_factorial = math.prod(range(2 * l - 1, 0, -2))
    return np.sqrt(
        2.0**l * (2.0 * alpha) ** (l + 1.5) / (math.pi**1.5 * double_factorial)
    )


def _molden_ao_order(basis) -> np.ndarray:
    """Forte2 AO index of each Molden AO, for the whole basis."""
    order = []
    offset = 0
    for ishell in range(basis.nshells):
        shell = basis[ishell]
        order.extend(offset + i for i in _molden_shell_order(shell.l))
        offset += shell.size
    return np.asarray(order, dtype=int)


def _molden_shell_order(l) -> list[int]:
    """Forte2 index (CCA, m = -l..+l) of each function in Molden order."""
    if l == 1:
        # Molden always writes p shells as (x, y, z) = (m=+1, m=-1, m=0).
        molden_m = [1, -1, 0]
    else:
        molden_m = [0]
        for m in range(1, l + 1):
            molden_m.extend((m, -m))
    index = {ml_from_shell_index_cca(l, i): i for i in range(2 * l + 1)}
    return [index[m] for m in molden_m]


def _atom_index(system, center) -> int:
    """1-based index of the atom at a basis-function center."""
    coords = np.array([np.asarray(c, dtype=float) for _, c in system.atoms])
    distances = np.linalg.norm(coords - np.asarray(center, dtype=float), axis=1)
    iatom = int(np.argmin(distances))
    if distances[iatom] > _CENTER_TOL:
        raise RuntimeError("A basis function center does not coincide with any atom.")
    return iatom + 1


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _atoms_section(system) -> list[str]:
    lines = ["[Atoms] AU"]
    for iatom, (Z, coords) in enumerate(system.atoms, start=1):
        x, y, z = (float(v) for v in coords)
        symbol = Z_TO_ATOM_SYMBOL[int(Z)]
        lines.append(
            f"{symbol:<2s} {iatom:4d} {int(Z):4d} {x: .14f} {y: .14f} {z: .14f}"
        )
    return lines + [""]


def _pure_shell_tags(basis) -> list[str]:
    max_l = max(basis[i].l for i in range(basis.nshells))
    tags = [f"[{2 * l + 1}{AM_LABELS[l].upper()}]" for l in range(2, max_l + 1)]
    return tags + [""] if tags else []


def _gto_section(system) -> list[str]:
    basis = system.basis
    lines = ["[GTO]"]
    for first, last in basis.center_first_and_last_shell:
        lines.append(f"{_atom_index(system, basis[first].center)} 0")
        for ishell in range(first, last):
            shell = basis[ishell]
            exponents = np.asarray(shell.exponents, dtype=float)
            # Forte2 coefficients include the primitive normalization; Molden
            # coefficients refer to normalized primitives.
            coefficients = np.asarray(shell.coeff, dtype=float) / _primitive_norm(
                shell.l, exponents
            )
            lines.append(f" {AM_LABELS[shell.l]:<2s} {shell.nprim:4d} 1.00")
            lines += [
                f"  {a: .10E}  {c: .10E}" for a, c in zip(exponents, coefficients)
            ]
        lines.append("")
    return lines


def _mo_sections(blocks, ao_order, split_spin_blocks) -> list[str]:
    lines = []
    for block in blocks:
        if not lines or split_spin_blocks:
            lines.append("[MO]")
        C = block.C[ao_order, :]
        for imo in range(C.shape[1]):
            lines += [
                f"Sym= {_ascii_line(block.symmetries[imo])}",
                f"Ene= {block.energies[imo]: .14f}",
                f"Spin= {block.spin}",
                f"Occup= {block.occupations[imo]: .14f}",
            ]
            lines += [f"{iao:5d} {c: .14E}" for iao, c in enumerate(C[:, imo], start=1)]
    return lines


def _ascii_line(text) -> str:
    return str(text).replace("\n", " ").encode("ascii", "replace").decode("ascii")
