"""Molden-format writer for molecular orbitals."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.system.basis_utils import AM_LABELS, ml_from_shell_index_cca
from .semicanonicalizer import Semicanonicalizer

__all__ = ["write_molden"]


@dataclass(frozen=True)
class _MoldenBlock:
    """Container for one Molden MO block."""

    coeff: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    sym_labels: list[str]
    spin: str


def write_molden(obj, path="orbitals.molden") -> None:
    """
    Write a Molden file for an RHF-, ROHF-, UHF-, or MCOpt-like object.

    Parameters
    ----------
    obj
        An object with Forte2-like orbital attributes. The writer requires
        ``system`` and ``C`` attributes, supports real-valued one- and
        two-spin-block references, and can also serialize final MCOpt orbitals.
    path : str or pathlib.Path, optional, default="orbitals.molden"
        Destination path for the Molden file.

    Raises
    ------
    TypeError
        If ``obj`` does not provide the required orbital interface.
    RuntimeError
        If orbital data is missing or inconsistent.
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

    system, mo_blocks = _extract_molden_blocks(obj)
    permutation = _molden_ao_permutation(system.basis)

    lines = ["[Molden Format]", ""]
    lines.extend(_format_atoms(system))
    lines.extend(_format_pure_shell_tags(system.basis))
    lines.extend(_format_gto(system.basis))
    lines.extend(_format_mo(mo_blocks, permutation))

    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _extract_molden_blocks(obj):
    system, coeff_blocks = _validate_molden_object(obj)

    if len(coeff_blocks) == 2:
        return system, _extract_uhf_mo_blocks(obj, coeff_blocks)

    if hasattr(obj, "mo_space") and callable(getattr(obj, "make_average_1rdm", None)):
        return system, [_extract_mcopt_mo_block(obj, coeff_blocks[0])]

    if _is_rohf_like(obj):
        return system, [_extract_rohf_mo_block(obj, coeff_blocks[0])]

    return system, [_extract_rhf_mo_block(obj, coeff_blocks[0])]


def _validate_molden_object(obj):
    for attr in ("system", "C"):
        if not hasattr(obj, attr):
            raise TypeError(
                "write_molden() requires an object with 'system' and 'C' attributes."
            )

    system = obj.system
    coeff_blocks = obj.C

    if system is None or coeff_blocks is None:
        raise RuntimeError("Orbital data is not available. Run the SCF method first.")

    if system.two_component:
        raise NotImplementedError("Two-component Molden output is not supported.")

    if len(coeff_blocks) not in (1, 2):
        raise NotImplementedError(
            "Only RHF/ROHF/UHF/MCOpt Molden output is supported."
        )

    coeff_arrays = []
    for coeff_block in coeff_blocks:
        coeff = np.asarray(coeff_block)
        if coeff.size == 0:
            raise RuntimeError("Orbital coefficients are missing.")
        if coeff.ndim != 2:
            raise RuntimeError("Orbital coefficients have invalid shapes.")
        if coeff.shape[0] != system.nbf:
            raise RuntimeError(
                "The MO coefficient matrix must be expressed in the AO basis."
            )
        if coeff.shape[1] != system.nmo:
            raise RuntimeError(
                "The MO coefficient matrix must span the full MO space."
            )
        if np.iscomplexobj(coeff):
            raise NotImplementedError("Complex-valued Molden output is not supported.")
        coeff_arrays.append(coeff)

    for ishell in range(system.basis.nshells):
        shell = system.basis[ishell]
        if not shell.is_pure:
            raise NotImplementedError("Cartesian-shell Molden output is not supported.")
        if shell.size != 2 * shell.l + 1:
            raise NotImplementedError(
                "General-contraction spherical shells are not supported in Molden output."
            )

    return system, coeff_arrays


def _extract_rhf_mo_block(obj, coeff: np.ndarray) -> _MoldenBlock:
    norb = coeff.shape[1]
    occupations = _filled_prefix_occupations(_get_ndocc(obj), norb, value=2.0)
    return _build_molden_block(obj, coeff, occupations)


def _extract_rohf_mo_block(obj, coeff: np.ndarray) -> _MoldenBlock:
    ndocc = getattr(obj, "ndocc", None)
    nsocc = getattr(obj, "nsocc", None)

    if ndocc is None or nsocc is None:
        na = getattr(obj, "na", None)
        nb = getattr(obj, "nb", None)
        if na is None or nb is None:
            raise RuntimeError(
                "ROHF occupation data is unavailable. Run the SCF method first."
            )
        ndocc = min(int(na), int(nb))
        nsocc = abs(int(na) - int(nb))

    occupations = np.zeros(coeff.shape[1], dtype=float)
    occupations[:ndocc] = 2.0
    occupations[ndocc : ndocc + nsocc] = 1.0
    return _build_molden_block(obj, coeff, occupations)


def _extract_uhf_mo_blocks(obj, coeff_blocks: list[np.ndarray]) -> list[_MoldenBlock]:
    norb = coeff_blocks[0].shape[1]
    energies = _get_energy_blocks(obj, 2, norb)
    occupations = [
        _filled_prefix_occupations(_get_electron_count(obj, "na"), norb, value=1.0),
        _filled_prefix_occupations(_get_electron_count(obj, "nb"), norb, value=1.0),
    ]
    spins = ["Alpha", "Beta"]

    blocks = []
    for ispin in range(2):
        blocks.append(
            _build_molden_block(
                obj,
                coeff_blocks[ispin],
                occupations[ispin],
                spin=spins[ispin],
                block_index=ispin,
                energies=energies[ispin],
            )
        )
    return blocks


def _extract_mcopt_mo_block(obj, coeff: np.ndarray) -> _MoldenBlock:
    norb = coeff.shape[1]
    occupations = np.zeros(norb, dtype=float)
    occupations[np.asarray(obj.mo_space.docc_indices, dtype=int)] = 2.0

    g1_act = _get_mcopt_active_density(obj)
    active_occ = np.real_if_close(np.diag(g1_act)).astype(float)
    occupations[np.asarray(obj.mo_space.active_indices, dtype=int)] = np.clip(
        active_occ, 0.0, 2.0
    )

    return _build_molden_block(
        obj,
        coeff,
        occupations,
        energies=_get_mcopt_energies(obj, g1_act, norb),
    )


def _get_mcopt_active_density(obj) -> np.ndarray:
    g1_act = np.asarray(obj.make_average_1rdm())
    if g1_act.ndim != 2 or g1_act.shape[0] != g1_act.shape[1]:
        raise RuntimeError("MCOpt active-space 1-RDM has an invalid shape.")
    if g1_act.shape[0] != obj.mo_space.nactv:
        raise RuntimeError("MCOpt active-space 1-RDM does not match the active space.")
    return g1_act


def _get_mcopt_energies(obj, g1_act: np.ndarray, norb: int) -> np.ndarray:
    orig_to_contig = np.asarray(obj.mo_space.orig_to_contig, dtype=int)
    final_orbital = getattr(obj, "final_orbital", "semicanonical")

    if final_orbital == "original":
        if not hasattr(obj, "orb_opt") or not hasattr(obj.orb_opt, "Fock"):
            raise RuntimeError(
                "MCOpt generalized Fock matrix is unavailable after optimization."
            )
        if not np.allclose(obj.C[0][:, orig_to_contig], obj.orb_opt.C):
            raise RuntimeError(
                "MCOpt generalized Fock matrix does not match the final orbital basis."
            )
        energies_contig = np.real_if_close(np.diag(np.asarray(obj.orb_opt.Fock))).astype(
            float
        )
    elif final_orbital == "semicanonical":
        semi = Semicanonicalizer(
            mo_space=obj.mo_space,
            system=obj.system,
            mix_inactive=False,
            mix_active=False,
        )
        C_contig = obj.C[0][:, orig_to_contig].copy()
        semi.semi_canonicalize(g1=g1_act, C_contig=C_contig)
        energies_contig = np.asarray(semi.eps_semican, dtype=float)
    else:
        raise NotImplementedError(
            f"Unsupported MCOpt final_orbital setting: {final_orbital!r}"
        )

    if energies_contig.shape != (norb,):
        raise RuntimeError("MCOpt generalized Fock diagonal has an invalid shape.")

    return energies_contig[orig_to_contig]


def _get_energy_blocks(
    obj, expected_blocks: int, norb: int, allow_missing: bool = False, default_value: float = 0.0
) -> list[np.ndarray]:
    eps = getattr(obj, "eps", None)
    if eps is None:
        if allow_missing:
            return [np.full(norb, default_value, dtype=float) for _ in range(expected_blocks)]
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
    obj, block_index: int, norb: int, allow_missing: bool = False, default_value: float = 0.0
) -> np.ndarray:
    return _get_energy_blocks(
        obj,
        expected_blocks=1,
        norb=norb,
        allow_missing=allow_missing,
        default_value=default_value,
    )[block_index]


def _build_molden_block(
    obj,
    coeff: np.ndarray,
    occupations: np.ndarray,
    *,
    spin: str = "Alpha",
    block_index: int = 0,
    energies: np.ndarray | None = None,
) -> _MoldenBlock:
    norb = coeff.shape[1]
    if energies is None:
        energies = _get_energy_block(obj, block_index, norb)
    return _MoldenBlock(
        coeff=coeff,
        energies=energies,
        occupations=occupations,
        sym_labels=_get_irrep_labels(obj, block_index, norb),
        spin=spin,
    )


def _get_irrep_labels(obj, block_index: int, norb: int) -> list[str]:
    irrep_labels = getattr(obj, "irrep_labels", None)
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


def _get_ndocc(obj) -> int:
    ndocc = getattr(obj, "ndocc", None)
    if ndocc is not None:
        return int(ndocc)

    na = getattr(obj, "na", None)
    nb = getattr(obj, "nb", None)
    if na is None or nb is None or na != nb:
        raise RuntimeError(
            "Restricted occupation data is unavailable. Run the method first."
        )
    return int(na)


def _get_electron_count(obj, attr: str) -> int:
    value = getattr(obj, attr, None)
    if value is None:
        raise RuntimeError(
            f"Spin-resolved occupation data '{attr}' is unavailable. Run the method first."
        )
    return int(value)


def _filled_prefix_occupations(nocc: int, norb: int, value: float) -> np.ndarray:
    occupations = np.zeros(norb, dtype=float)
    occupations[:nocc] = value
    return occupations


def _is_rohf_like(obj) -> bool:
    return hasattr(obj, "nsocc") or (
        hasattr(obj, "na")
        and hasattr(obj, "nb")
        and getattr(obj, "na") is not None
        and getattr(obj, "nb") is not None
        and int(getattr(obj, "na")) != int(getattr(obj, "nb"))
    )


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
            f"{symbol:<2s} {iatom:4d} {int(Z):4d} "
            f"{x: .14f} {y: .14f} {z: .14f}"
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
            for exponent, coeff in zip(shell.exponents, shell.coeff):
                lines.append(f"  {exponent: .10E}  {coeff: .10E}")
        lines.append("")
    return lines


def _format_mo(mo_blocks: list[_MoldenBlock], permutation: np.ndarray) -> list[str]:
    lines = ["[MO]"]
    for block in mo_blocks:
        coeff = block.coeff[permutation, :]
        for imo in range(coeff.shape[1]):
            lines.append(f"Sym= {block.sym_labels[imo]}")
            lines.append(f"Ene= {block.energies[imo]: .14f}")
            lines.append(f"Spin= {block.spin}")
            lines.append(f"Occup= {block.occupations[imo]: .14f}")
            for iao, value in enumerate(coeff[:, imo], start=1):
                lines.append(f"{iao:4d} {value: .14E}")
    return lines
