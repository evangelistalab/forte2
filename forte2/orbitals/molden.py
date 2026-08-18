"""Molden-format writer for molecular orbitals."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forte2.base_classes import Method
from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.system.basis_utils import AM_LABELS, ml_from_shell_index_cca
from .semicanonicalizer import Semicanonicalizer

__all__ = ["write_molden"]


@dataclass(frozen=True)
class _MoldenBlock:
    """Holds information for one block of MOs."""

    C: np.ndarray
    eps: np.ndarray
    occ: np.ndarray
    sym_labels: list[str]
    spin: str


def write_molden(method: Method, path: str | Path = "orbitals.molden") -> None:
    """
    Write a Molden file for any method that provides orbitals

    Parameters
    ----------
    method : Method
        A Forte2 method that provides ``system`` and ``mos``. The writer
        supports real-valued one- and two-spin-block references and can also
        serialize final MCOptimizer orbitals.
    path : str or pathlib.Path, optional, default="orbitals.molden"
        Destination path for the Molden file.

    Raises
    ------
    TypeError
        If ``method`` is not a :class:`~forte2.base_classes.Method`.
    RuntimeError
        If ``method`` does not provide the required data or its orbital data
        is missing or inconsistent.
    NotImplementedError
        If the object uses an unsupported reference or basis representation,
        such as two-component, complex-valued, or cartesian-shell orbitals.

    Notes
    -----
    Forte2 stores spherical AO coefficients in the Libint/CCA order
    :math:`m=-l, \\ldots, 0, \\ldots, +l`. Molden expects the spherical order
    :math:`0, +1, -1, +2, -2, \\ldots`. This function derives the per-shell
    permutation from the angular-momentum metadata in
    :mod:`forte2.system.basis_utils` and applies it only when writing the
    ``[MO]`` section.
    """

    requires = {"system", "mos"}
    _validate_molden_method(method, requires)

    system, mo_blocks = _extract_molden_blocks(method)
    permutation = _molden_ao_permutation(system.basis)

    lines = ["[Molden Format]", ""]
    lines.extend(_format_atoms(system))
    lines.extend(_format_pure_shell_tags(system.basis))
    lines.extend(_format_gto(system.basis))
    lines.extend(_format_mo(mo_blocks, permutation))

    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _validate_molden_method(method: Method, requires: set[str]) -> None:
    if not isinstance(method, Method):
        raise TypeError(
            "write_molden() requires 'method' to be an instance of "
            "forte2.base_classes.Method."
        )

    missing = requires - method.provides
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Method {method.__class__.__name__} does not provide required data: "
            f"{missing_str}."
        )


def _extract_molden_blocks(method):
    system, C_blocks = _validate_molden_data(method)

    if len(C_blocks) == 2:
        return system, _extract_uhf_mo_blocks(method, C_blocks)

    if hasattr(method, "mo_space") and callable(
        getattr(method, "make_average_1rdm", None)
    ):
        return system, [_extract_mcopt_mo_block(method, C_blocks[0])]

    return system, [_make_rhf_mo_block(method, C_blocks[0])]


def _validate_molden_data(method):
    system = getattr(method, "system", None)
    mos = getattr(method, "mos", None)
    C_blocks = getattr(mos, "C", None)

    if system is None or C_blocks is None:
        raise RuntimeError("Orbital data is not available. Run the method first.")

    if system.two_component:
        raise NotImplementedError("Two-component Molden output is not supported.")

    if len(C_blocks) not in (1, 2):
        raise NotImplementedError(
            "Only RHF/ROHF/UHF/MCOptimizer Molden output is supported."
        )

    C_arrays = []
    for C_block in C_blocks:
        C = np.asarray(C_block)
        if C.size == 0:
            raise RuntimeError("Orbital coefficients are missing.")
        if C.ndim != 2:
            raise RuntimeError("Orbital coefficients have invalid shapes.")
        if C.shape[0] != system.nbf:
            raise RuntimeError(
                "The MO coefficient matrix must be expressed in the AO basis."
            )
        if C.shape[1] != system.nmo:
            raise RuntimeError("The MO coefficient matrix must span the full MO space.")
        if np.iscomplexobj(C):
            raise NotImplementedError("Complex-valued Molden output is not supported.")
        C_arrays.append(C)

    for ishell in range(system.basis.nshells):
        shell = system.basis[ishell]
        if not shell.is_pure:
            raise NotImplementedError("Cartesian-shell Molden output is not supported.")
        if shell.size != 2 * shell.l + 1:
            raise NotImplementedError(
                "General-contraction spherical shells are not supported in Molden output."
            )

    return system, C_arrays


def _make_rhf_mo_block(method, C: np.ndarray) -> _MoldenBlock:
    norb = C.shape[1]
    ndocc, nsocc, _, _ = _get_scf_occupation(method)
    occupations = np.zeros(C.shape[1], dtype=float)
    occupations[:ndocc] = 2.0
    occupations[ndocc : ndocc + nsocc] = 1.0
    return _build_molden_block(method, C, occupations)


def _extract_uhf_mo_blocks(method, C_blocks: list[np.ndarray]) -> list[_MoldenBlock]:
    norb = C_blocks[0].shape[1]
    energies = _get_energy_blocks(method, 2, norb)
    _, _, na, nb = _get_scf_occupation(method)
    occupations = [
        _filled_prefix_occupations(na, norb, value=1.0),
        _filled_prefix_occupations(nb, norb, value=1.0),
    ]
    spins = ["Alpha", "Beta"]

    blocks = []
    for ispin in range(2):
        blocks.append(
            _build_molden_block(
                method,
                C_blocks[ispin],
                occupations[ispin],
                spin=spins[ispin],
                block_index=ispin,
                energies=energies[ispin],
            )
        )
    return blocks


def _extract_mcopt_mo_block(method, C: np.ndarray) -> _MoldenBlock:
    norb = C.shape[1]
    occupations = np.zeros(norb, dtype=float)
    occupations[np.asarray(method.mo_space.docc_indices, dtype=int)] = 2.0

    g1_act = _get_mcopt_active_density(method)
    active_occ = np.real_if_close(np.diag(g1_act)).astype(float)
    occupations[np.asarray(method.mo_space.active_indices, dtype=int)] = np.clip(
        active_occ, 0.0, 2.0
    )

    return _build_molden_block(
        method,
        C,
        occupations,
        energies=_get_mcopt_energies(method, C, g1_act, norb),
    )


def _get_mcopt_active_density(method) -> np.ndarray:
    g1_act = np.asarray(method.make_average_1rdm())
    if g1_act.ndim != 2 or g1_act.shape[0] != g1_act.shape[1]:
        raise RuntimeError("MCOptimizer active-space 1-RDM has an invalid shape.")
    if g1_act.shape[0] != method.mo_space.nactv:
        raise RuntimeError(
            "MCOptimizer active-space 1-RDM does not match the active space."
        )
    return g1_act


def _get_mcopt_energies(
    method, C: np.ndarray, g1_act: np.ndarray, norb: int
) -> np.ndarray:
    orig_to_contig = np.asarray(method.mo_space.orig_to_contig, dtype=int)
    contig_to_orig = np.asarray(method.mo_space.contig_to_orig, dtype=int)
    final_orbitals = getattr(method, "final_orbitals", "semicanonical")

    if final_orbitals == "original":
        if not hasattr(method, "orb_opt") or not hasattr(method.orb_opt, "Fock"):
            raise RuntimeError(
                "MCOptimizer generalized Fock matrix is unavailable after optimization."
            )
        if not np.allclose(C[:, orig_to_contig], method.orb_opt.C):
            raise RuntimeError(
                "MCOptimizer generalized Fock matrix does not match the final orbital basis."
            )
        energies_contig = np.real_if_close(
            np.diag(np.asarray(method.orb_opt.Fock))
        ).astype(float)
    elif final_orbitals in ("semicanonical", "natural"):
        irrep_indices = np.asarray(method.mos.irrep_indices[0], dtype=int)[
            orig_to_contig
        ]
        semi = Semicanonicalizer(
            mo_space=method.mo_space,
            system=method.system,
            irrep_indices=irrep_indices,
            mix_inactive=False,
            mix_active=False,
            do_active=(final_orbitals == "semicanonical"),
        )
        C_contig = C[:, orig_to_contig].copy()
        semi.semi_canonicalize(g1=g1_act, C_contig=C_contig)
        if final_orbitals == "semicanonical":
            energies_contig = np.asarray(semi.eps_semican, dtype=float)
        else:
            # Natural active orbitals do not diagonalize the generalized Fock
            # matrix, so report its diagonal in the actual final MO basis.
            energies_contig = np.real_if_close(np.diag(semi.fock)).astype(float)
    else:
        raise NotImplementedError(
            f"Unsupported MCOptimizer final_orbitals setting: {final_orbitals!r}"
        )

    if energies_contig.shape != (norb,):
        raise RuntimeError(
            "MCOptimizer generalized Fock diagonal has an invalid shape."
        )

    return energies_contig[contig_to_orig]


def _get_energy_blocks(
    method,
    expected_blocks: int,
    norb: int,
    allow_missing: bool = False,
    default_value: float = 0.0,
) -> list[np.ndarray]:
    eps = getattr(method, "eps", None)
    if eps is None:
        if allow_missing:
            return [
                np.full(norb, default_value, dtype=float)
                for _ in range(expected_blocks)
            ]
        raise RuntimeError("Orbital energies are unavailable. Run the method first.")

    if len(eps) != expected_blocks:
        raise NotImplementedError(
            "The number of orbital-energy blocks is incompatible with the Molden writer."
        )

    energies = []
    for block in eps:
        energy = np.asarray(block)
        if energy.ndim != 1 or energy.shape[0] != norb:
            raise RuntimeError("Orbital energies have invalid shapes.")
        if np.iscomplexobj(energy):
            raise NotImplementedError("Complex-valued Molden output is not supported.")
        energies.append(energy.astype(float, copy=False))
    return energies


def _get_energy_block(
    method,
    block_index: int,
    norb: int,
    allow_missing: bool = False,
    default_value: float = 0.0,
) -> np.ndarray:
    return _get_energy_blocks(
        method,
        expected_blocks=1,
        norb=norb,
        allow_missing=allow_missing,
        default_value=default_value,
    )[block_index]


def _build_molden_block(
    method,
    C: np.ndarray,
    occupations: np.ndarray,
    *,
    spin: str = "Alpha",
    block_index: int = 0,
    energies: np.ndarray | None = None,
) -> _MoldenBlock:
    norb = C.shape[1]
    if energies is None:
        energies = _get_energy_block(method, block_index, norb)
    return _MoldenBlock(
        C=C,
        eps=energies,
        occ=occupations,
        sym_labels=_get_irrep_labels(method, block_index, norb),
        spin=spin,
    )


def _get_irrep_labels(method, block_index: int, norb: int) -> list[str]:
    mos = getattr(method, "mos", None)
    irrep_labels = getattr(mos, "irrep_labels", None)
    if irrep_labels is None:
        irrep_labels = getattr(method, "irrep_labels", None)
    if irrep_labels is None:
        return ["a"] * norb

    labels = irrep_labels
    if len(labels) > 0 and isinstance(labels[0], (list, tuple, np.ndarray)):
        if block_index >= len(labels):
            return ["a"] * norb
        labels = labels[block_index]

    if len(labels) != norb:
        return ["a"] * norb
    return [str(label) for label in labels]


def _get_scf_occupation(method) -> int:
    na = getattr(method, "na", None)
    nb = getattr(method, "nb", None)
    if na is None or nb is None:
        raise RuntimeError(
            "Electron number data is unavailable. Run the SCF method first."
        )
    ndocc = min(int(na), int(nb))
    nsocc = abs(int(na) - int(nb))
    return ndocc, nsocc, na, nb


def _filled_prefix_occupations(nocc: int, norb: int, value: float) -> np.ndarray:
    occupations = np.zeros(norb, dtype=float)
    occupations[:nocc] = value
    return occupations


def _molden_ao_permutation(basis) -> np.ndarray:
    permutation = []
    offset = 0
    for ishell in range(basis.nshells):
        shell = basis[ishell]
        shell_perm = _molden_shell_permutation(shell.l, shell.size)
        permutation.extend(offset + idx for idx in shell_perm)
        offset += shell.size

    if offset != basis.size:
        raise RuntimeError("The AO permutation does not cover the full basis.")

    return np.asarray(permutation, dtype=int)


def _molden_shell_permutation(l: int, shell_size: int) -> list[int]:
    if shell_size != 2 * l + 1:
        raise NotImplementedError(
            "Only pure spherical shells with size 2*l+1 are supported."
        )

    m_to_internal_index = {
        ml_from_shell_index_cca(l, idx): idx for idx in range(shell_size)
    }
    molden_m_sequence = [0]
    for m in range(1, l + 1):
        molden_m_sequence.extend((m, -m))
    return [m_to_internal_index[m] for m in molden_m_sequence]


def _format_atoms(system) -> list[str]:
    lines = ["[Atoms] AU"]
    for iatom, (Z, coords) in enumerate(system.atoms, start=1):
        symbol = Z_TO_ATOM_SYMBOL[int(Z)]
        x, y, z = coords
        lines.append(
            f"{symbol:<2s} {iatom:4d} {int(Z):4d} " f"{x: .14f} {y: .14f} {z: .14f}"
        )
    lines.append("")
    return lines


def _format_pure_shell_tags(basis) -> list[str]:
    max_l = max(basis[ishell].l for ishell in range(basis.nshells))
    if max_l < 2:
        return []

    lines = []
    for l in range(2, max_l + 1):
        lines.append(f"[{2 * l + 1}{AM_LABELS[l].upper()}]")
    lines.append("")
    return lines


def _format_gto(basis) -> list[str]:
    lines = ["[GTO]"]
    for icenter, (shell_first, shell_last) in enumerate(
        basis.center_first_and_last_shell, start=1
    ):
        lines.append(f"{icenter} 0")
        for ishell in range(shell_first, shell_last):
            shell = basis[ishell]
            shell_label = AM_LABELS[shell.l]
            lines.append(f" {shell_label:<2s} {shell.nprim:4d} 1.00")
            for exponent, C_primitive in zip(shell.exponents, shell.coeff):
                lines.append(f"  {exponent: .10E}  {C_primitive: .10E}")
        lines.append("")
    return lines


def _format_mo(mo_blocks: list[_MoldenBlock], permutation: np.ndarray) -> list[str]:
    lines = ["[MO]"]
    for block in mo_blocks:
        C = block.C[permutation, :]
        for imo in range(C.shape[1]):
            lines.append(f"Sym= {block.sym_labels[imo]}")
            lines.append(f"Ene= {block.eps[imo]: .14f}")
            lines.append(f"Spin= {block.spin}")
            lines.append(f"Occup= {block.occ[imo]: .14f}")
            for iao, value in enumerate(C[:, imo], start=1):
                lines.append(f"{iao:4d} {value: .14E}")
    return lines
