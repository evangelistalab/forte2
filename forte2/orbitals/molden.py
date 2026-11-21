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
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forte2.base_classes import Method
from forte2.base_classes.active_space_driver import ActiveSpaceDriver
from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.system import System
from forte2.system.basis_utils import AM_LABELS, ml_from_shell_index_cca

__all__ = ["write_molden", "molden_string"]

MAX_MOLDEN_L = 4
"""The Molden format defines shells up to g."""

_ZERO_THRESHOLD = 1e-12


@dataclass(frozen=True)
class _MoldenBlock:
    """One spin block of MOs, in Forte2's AO order."""

    C: np.ndarray
    eps: np.ndarray
    occ: np.ndarray
    sym_labels: list[str]
    spin: str


def write_molden(source, path="orbitals.molden", **kwargs) -> Path:
    """
    Write molecular orbitals to a Molden file.

    Parameters
    ----------
    source : Method or System
        An executed Forte2 method that provides ``system`` and ``mos`` (SCF
        references, orbital-rotation methods such as AVAS or ASET, and
        active-space drivers such as CI and MCOptimizer), or a ``System``
        together with ``C=...``.
    path : str or pathlib.Path, optional, default="orbitals.molden"
        Destination path.
    **kwargs
        Passed to :func:`molden_string` (``C``, ``energies``, ``occupations``,
        ``symmetries``, ``title``).

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
) -> str:
    """
    Return molecular orbitals as Molden-format text.

    Parameters
    ----------
    source : Method or System
        An executed Forte2 method, or a ``System``. See :func:`write_molden`.
    C : ndarray or list of ndarray, optional
        Required with a ``System`` and not allowed with a method. Real AO-basis
        coefficients, shape ``(nbf, nmo)``. Pass one array for restricted
        orbitals or ``[C_alpha, C_beta]`` for unrestricted ones.
    energies, occupations : ndarray or list of ndarray, optional
        Per-orbital values with the same block structure as ``C``. Default to
        zero.
    symmetries : list of str or list of list of str, optional
        Per-orbital symmetry labels. Default to ``"A"``.
    title : str, optional
        Text of the ``[Title]`` section.

    Returns
    -------
    str
        The Molden file contents.

    Raises
    ------
    TypeError
        If ``source`` is neither a method nor a system, or if orbital arrays
        are combined with a method.
    RuntimeError
        If the method has not been run or does not provide orbitals.
    NotImplementedError
        For two-component or complex orbitals, Cartesian or generally
        contracted shells, or shells beyond g.
    """
    if isinstance(source, Method):
        if any(x is not None for x in (C, energies, occupations, symmetries)):
            raise TypeError(
                "C, energies, occupations and symmetries can only be combined with a "
                "System. To write other orbitals of a method, pass method.system."
            )
        system, blocks = _blocks_from_method(source)
        default_title = f"Forte2 {type(source).__name__} orbitals"
    elif isinstance(source, System):
        if C is None:
            raise TypeError("C is required when writing the orbitals of a System.")
        system = source
        blocks = _blocks_from_arrays(system, C, energies, occupations, symmetries)
        default_title = "Forte2 orbitals"
    else:
        raise TypeError(
            "molden_string() expects a forte2 Method or System, "
            f"got {type(source).__name__}."
        )

    _check_basis(system)
    lines = ["[Molden Format]", "[Title]", _ascii_line(title or default_title), ""]
    lines.extend(_format_atoms(system))
    lines.extend(_format_pure_shell_tags(system.basis))
    lines.extend(_format_gto(system))
    lines.extend(_format_mo(blocks, _molden_ao_permutation(system.basis)))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orbital data
# --------------------------------------------------------------------------- #


def _blocks_from_method(method):
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

    system = method.system
    mos = method.mos
    # Use the method's own flag: System.two_component is mutated in place when
    # a shared System is bound to another (e.g. GHF) method.
    if method.two_component or mos.spinorbital:
        raise NotImplementedError(
            "Two-component orbitals cannot be written in Molden format. Use "
            "forte2.write_orbital_cubes(..., format=('2ccube',)) instead."
        )
    C_blocks = [_real_coefficients(C, system) for C in mos.C]
    nmo = C_blocks[0].shape[1]
    labels = [_symmetry_labels(lab, nmo) for lab in mos.irrep_labels]

    if isinstance(method, ActiveSpaceDriver):
        _require_single_block(name, C_blocks)
        occ, eps = _active_space_driver_data(method, C_blocks[0])
        return system, [_MoldenBlock(C_blocks[0], eps, occ, labels[0], "Alpha")]

    if "mo_space" in method.provides:
        # Orbital-rotation methods (AVAS, ASET, ...): nominal occupations and
        # orbital energies come from the SCF reference they rotate.
        _require_single_block(name, C_blocks)
        occ, eps = _reference_projected_data(method, C_blocks[0])
        return system, [_MoldenBlock(C_blocks[0], eps, occ, labels[0], "Alpha")]

    scf = _as_scf(method)
    if scf is None:
        raise NotImplementedError(
            f"write_molden() does not know the occupations of {name} orbitals. "
            "Pass method.system with C=..., energies=..., occupations=... instead."
        )
    eps_blocks = _energy_blocks(scf, len(C_blocks), nmo)
    occ_blocks = _scf_occupations(scf, system, C_blocks)
    spins = ["Alpha"] if len(C_blocks) == 1 else ["Alpha", "Beta"]
    return system, [
        _MoldenBlock(C, eps, occ, lab, spin)
        for C, eps, occ, lab, spin in zip(
            C_blocks, eps_blocks, occ_blocks, labels, spins
        )
    ]


def _blocks_from_arrays(system, C, energies, occupations, symmetries):
    C_blocks = [C] if isinstance(C, np.ndarray) and C.ndim == 2 else list(C)
    if len(C_blocks) not in (1, 2):
        raise ValueError("C must be one coefficient matrix or [C_alpha, C_beta].")
    C_blocks = [_real_coefficients(c, system, full_space=False) for c in C_blocks]
    nmo = C_blocks[0].shape[1]
    if any(c.shape[1] != nmo for c in C_blocks):
        raise ValueError(
            "Alpha and beta coefficient matrices must have the same shape."
        )

    def per_block(values, default):
        if values is None:
            return [np.full(nmo, default, dtype=float) for _ in C_blocks]
        vals = [values] if np.ndim(values[0]) == 0 else list(values)
        if len(vals) != len(C_blocks):
            raise ValueError(
                "Per-orbital values must have one block per coefficient matrix."
            )
        out = []
        for v in vals:
            v = np.asarray(v, dtype=float)
            if v.shape != (nmo,):
                raise ValueError(
                    f"Expected {nmo} per-orbital values, got shape {v.shape}."
                )
            out.append(v)
        return out

    eps_blocks = per_block(energies, 0.0)
    occ_blocks = per_block(occupations, 0.0)
    if symmetries is None:
        sym_blocks = [["A"] * nmo for _ in C_blocks]
    elif isinstance(symmetries[0], str):
        sym_blocks = [list(symmetries)]
    else:
        sym_blocks = [list(s) for s in symmetries]
    if len(sym_blocks) != len(C_blocks) or any(len(s) != nmo for s in sym_blocks):
        raise ValueError("symmetries must provide one label per orbital and block.")
    spins = ["Alpha"] if len(C_blocks) == 1 else ["Alpha", "Beta"]
    return [
        _MoldenBlock(c, e, o, [str(x) for x in s], spin)
        for c, e, o, s, spin in zip(C_blocks, eps_blocks, occ_blocks, sym_blocks, spins)
    ]


def _real_coefficients(C, system, full_space=True):
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
    if full_space and C.shape[1] != system.nmo:
        raise RuntimeError("The MO coefficient matrix must span the full MO space.")
    if np.iscomplexobj(C):
        if np.abs(C.imag).max() > 1e-10:
            raise NotImplementedError(
                "Complex-valued orbitals cannot be written in Molden format."
            )
        C = C.real
    return np.asarray(C, dtype=float)


def _require_single_block(name, C_blocks):
    if len(C_blocks) != 1:
        raise NotImplementedError(
            f"write_molden() supports {name} only with restricted orbitals."
        )


def _symmetry_labels(labels, nmo):
    labels = [str(x) for x in labels] if labels is not None else []
    return labels if len(labels) == nmo else ["A"] * nmo


def _as_scf(method):
    """Return ``method`` if it carries SCF densities and electron counts."""
    if all(hasattr(method, a) for a in ("D", "F", "eps", "na", "nb")):
        return method
    return None


def _reference_scf(method):
    """Walk up the method chain to the SCF reference that defines occupations."""
    stage = getattr(method, "parent_method", None)
    while stage is not None:
        if _as_scf(stage) is not None:
            return stage
        stage = getattr(stage, "parent_method", None)
    raise NotImplementedError(
        f"Could not find the SCF reference of {type(method).__name__} to define "
        "orbital occupations. Pass method.system with C=... and occupations=... instead."
    )


def _spin_densities(scf):
    """Return ``[D_alpha, D_beta]`` in the AO basis (RHF stores only D_alpha)."""
    D = [np.asarray(d) for d in scf.D]
    return D * 2 if len(D) == 1 else D


def _project(C, S, D):
    """Occupations n_i = c_i^T S D S c_i of orbitals C in density D."""
    SC = S @ C
    occ = np.einsum("mi,mn,ni->i", SC, D, SC).real
    occ[np.abs(occ) < 1e-10] = 0.0
    return occ


def _overlap(system):
    from forte2.integrals import overlap

    return np.asarray(overlap(system))


def _scf_occupations(scf, system, C_blocks):
    S = _overlap(system)
    Da, Db = _spin_densities(scf)
    if len(C_blocks) == 1:
        return [_project(C_blocks[0], S, Da) + _project(C_blocks[0], S, Db)]
    return [_project(C_blocks[0], S, Da), _project(C_blocks[1], S, Db)]


def _energy_blocks(scf, nblocks, nmo):
    eps = list(scf.eps)
    if len(eps) != nblocks:
        raise NotImplementedError(
            "The number of orbital-energy blocks does not match the orbitals."
        )
    out = []
    for e in eps:
        e = np.real_if_close(np.asarray(e))
        if e.shape != (nmo,):
            raise RuntimeError("Orbital energies have invalid shapes.")
        out.append(np.asarray(e, dtype=float))
    return out


def _reference_projected_data(method, C):
    """Nominal occupations and Fock diagonal of rotated reference orbitals."""
    scf = _reference_scf(method)
    if len(scf.F) != 1:
        raise NotImplementedError(
            f"write_molden() supports {type(method).__name__} only on a restricted "
            "(RHF or ROHF) reference."
        )
    S = _overlap(method.system)
    Da, Db = _spin_densities(scf)
    occ = np.clip(_project(C, S, Da) + _project(C, S, Db), 0.0, 2.0)
    F = np.asarray(scf.F[0])
    eps = np.real_if_close(np.einsum("mi,mn,ni->i", C, F, C)).astype(float)
    return occ, eps


def _active_space_driver_data(method, C):
    """Occupations and generalized-Fock diagonal for CI/MCSCF drivers.

    Core orbitals are doubly occupied, active orbitals carry the diagonal of the
    state-averaged one-particle density matrix (natural occupation numbers when
    ``final_orbitals="natural"``), virtual orbitals are empty. Energies are the
    diagonal of the generalized Fock matrix in the final orbital basis, which
    equals the semicanonical orbital energies when
    ``final_orbitals="semicanonical"``.
    """
    from .semicanonicalizer import Semicanonicalizer

    mo_space = method.mo_space
    nmo = C.shape[1]
    g1 = np.asarray(method.make_average_rdm(1))
    if g1.shape != (mo_space.nactv, mo_space.nactv):
        raise RuntimeError("The active-space 1-RDM does not match the active space.")

    occ = np.zeros(nmo)
    occ[np.asarray(mo_space.docc_indices, dtype=int)] = 2.0
    occ[np.asarray(mo_space.active_indices, dtype=int)] = np.clip(
        np.real_if_close(np.diag(g1)).astype(float), 0.0, 2.0
    )

    orig_to_contig = np.asarray(mo_space.orig_to_contig, dtype=int)
    contig_to_orig = np.asarray(mo_space.contig_to_orig, dtype=int)
    semi = Semicanonicalizer(
        mo_space=mo_space,
        system=method.system,
        irrep_indices=np.asarray(method.mos.irrep_indices[0], dtype=int)[
            orig_to_contig
        ],
        mix_inactive=False,
        mix_active=False,
        do_active=False,
    )
    semi.semi_canonicalize(g1=g1, C_contig=C[:, orig_to_contig].copy())
    eps_contig = np.real_if_close(np.diag(semi.fock)).astype(float)
    if eps_contig.shape != (nmo,):
        raise RuntimeError("The generalized Fock matrix has an invalid shape.")
    return occ, eps_contig[contig_to_orig]


# --------------------------------------------------------------------------- #
# Basis and formatting
# --------------------------------------------------------------------------- #


def _check_basis(system):
    basis = system.basis
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


def _double_factorial(n):
    return math.prod(range(n, 0, -2)) if n > 0 else 1


def _primitive_norm(l, alpha):
    """Normalization of the axial Cartesian primitive x^l exp(-alpha r^2).

    This is the factor Libint (and therefore Forte2) folds into shell
    coefficients, and the factor Molden readers apply to each primitive.
    """
    alpha = np.asarray(alpha, dtype=float)
    return np.sqrt(
        2.0**l
        * (2.0 * alpha) ** (l + 1.5)
        / (math.pi**1.5 * _double_factorial(2 * l - 1))
    )


def _molden_ao_permutation(basis) -> np.ndarray:
    """Index array mapping Molden AO order to Forte2 AO order."""
    permutation = []
    offset = 0
    for ishell in range(basis.nshells):
        shell = basis[ishell]
        permutation.extend(offset + i for i in _molden_shell_permutation(shell.l))
        offset += shell.size
    if offset != basis.size:
        raise RuntimeError("The AO permutation does not cover the full basis.")
    return np.asarray(permutation, dtype=int)


def _molden_shell_permutation(l: int) -> list[int]:
    m_to_index = {ml_from_shell_index_cca(l, i): i for i in range(2 * l + 1)}
    if l == 1:
        # Molden always writes p shells as (x, y, z) = (m=+1, m=-1, m=0).
        molden_m = [1, -1, 0]
    else:
        molden_m = [0]
        for m in range(1, l + 1):
            molden_m.extend((m, -m))
    return [m_to_index[m] for m in molden_m]


def _shell_atom_indices(system):
    """Map each basis center to the 1-based index of the atom it sits on."""
    coords = np.array([np.asarray(c, dtype=float) for _, c in system.atoms])
    out = []
    for first, _last in system.basis.center_first_and_last_shell:
        center = np.asarray(system.basis[first].center, dtype=float)
        d = np.linalg.norm(coords - center, axis=1)
        iatom = int(np.argmin(d))
        if d[iatom] > 1e-6:
            raise RuntimeError(
                "A basis function center does not coincide with any atom."
            )
        out.append(iatom + 1)
    return out


def _format_atoms(system) -> list[str]:
    lines = ["[Atoms] AU"]
    for iatom, (Z, coords) in enumerate(system.atoms, start=1):
        symbol = Z_TO_ATOM_SYMBOL[int(Z)]
        x, y, z = (float(v) for v in coords)
        lines.append(
            f"{symbol:<2s} {iatom:4d} {int(Z):4d} {x: .14f} {y: .14f} {z: .14f}"
        )
    lines.append("")
    return lines


def _format_pure_shell_tags(basis) -> list[str]:
    max_l = max(basis[i].l for i in range(basis.nshells))
    tags = [f"[{2 * l + 1}{AM_LABELS[l].upper()}]" for l in range(2, max_l + 1)]
    return tags + [""] if tags else []


def _format_gto(system) -> list[str]:
    basis = system.basis
    lines = ["[GTO]"]
    for (first, last), iatom in zip(
        basis.center_first_and_last_shell, _shell_atom_indices(system)
    ):
        lines.append(f"{iatom} 0")
        for ishell in range(first, last):
            shell = basis[ishell]
            exponents = np.asarray(shell.exponents, dtype=float)
            # Forte2 coefficients include the primitive normalization; Molden
            # coefficients refer to normalized primitives.
            coefficients = np.asarray(shell.coeff, dtype=float) / _primitive_norm(
                shell.l, exponents
            )
            lines.append(f" {AM_LABELS[shell.l]:<2s} {shell.nprim:4d} 1.00")
            for exponent, c in zip(exponents, coefficients):
                lines.append(f"  {exponent: .10E}  {c: .10E}")
        lines.append("")
    return lines


def _format_mo(blocks, permutation) -> list[str]:
    lines = ["[MO]"]
    for block in blocks:
        C = block.C[permutation, :]
        for imo in range(C.shape[1]):
            lines.append(f"Sym= {_ascii_line(block.sym_labels[imo])}")
            lines.append(f"Ene= {block.eps[imo]: .14f}")
            lines.append(f"Spin= {block.spin}")
            lines.append(f"Occup= {block.occ[imo]: .14f}")
            for iao, value in enumerate(C[:, imo], start=1):
                lines.append(f"{iao:5d} {value: .14E}")
    return lines


def _ascii_line(text) -> str:
    return str(text).replace("\n", " ").encode("ascii", "replace").decode("ascii")
