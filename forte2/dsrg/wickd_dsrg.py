"""Wick&d-generated single-reference DSRG(n) equations."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

try:
    import wickd as w
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional dep
    w = None

from forte2 import RHF, System
from forte2.helpers import DIIS
from forte2.lib.det import Determinant

SCREEN = 1.0e-12


@dataclass(frozen=True)
class WickdDSRGData:
    """Spatial-orbital data for closed-shell single-reference DSRG."""

    scalar: float
    hcore_mo: np.ndarray
    eri_mo: np.ndarray
    eps: np.ndarray
    nocc: int
    reference: Determinant | None = None

    @property
    def nmo(self) -> int:
        return int(self.hcore_mo.shape[0])


@dataclass(frozen=True)
class WickdCommutator:
    """Compiled Wick&d commutator equations."""

    rank: int
    functions: tuple
    blocks: tuple[str, ...]
    generation_s: dict[str, float]
    n_equations: int
    n_expression_terms: int


@dataclass(frozen=True)
class WickdDSRGIteration:
    """One Wick&d DSRG fixed-point iteration."""

    iteration: int
    energy: float
    delta_energy: float
    rms_update: float
    ncomm: int
    iter_s: float


@dataclass(frozen=True)
class WickdDSRGResult:
    """Result of a Wick&d-generated DSRG(n) solve."""

    energy: float
    converged: bool
    iterations: int
    max_rank: int
    amplitudes: np.ndarray
    history: tuple[WickdDSRGIteration, ...]
    hbar: dict[str, np.ndarray]
    equations: WickdCommutator
    seconds: float


def _require_wickd():
    if w is None:
        raise ModuleNotFoundError("wickd is required for WickdDSRG")


def wickd_dsrg_data_from_rhf(rhf: RHF) -> WickdDSRGData:
    """Build Wick&d DSRG input data from a converged restricted HF object."""
    system = rhf.system
    if not isinstance(system, System):
        raise TypeError("rhf.system must be a forte2.System")
    if rhf.na != rhf.nb:
        raise ValueError("Wick&d DSRG currently requires a closed-shell RHF reference")

    coeff = rhf.C[0]
    hcore_mo = np.einsum(
        "pq,pi,qj->ij", system.ints_hcore(), coeff, coeff, optimize=True
    )
    eri_mo = system.fock_builder.two_electron_integrals_block(coeff)
    reference = Determinant("2" * rhf.na + "0" * (rhf.nmo - rhf.na))
    return WickdDSRGData(
        scalar=float(system.nuclear_repulsion),
        hcore_mo=np.asarray(hcore_mo, dtype=float),
        eri_mo=np.asarray(eri_mo, dtype=float),
        eps=np.asarray(rhf.eps[0], dtype=float),
        nocc=int(rhf.na),
        reference=reference,
    )


def spin_modes(nspatial: int, nocc_spatial: int):
    occ = [(i, spin) for i in range(nocc_spatial) for spin in ("a", "b")]
    virt = [(a, spin) for a in range(nocc_spatial, nspatial) for spin in ("a", "b")]
    return occ, virt


def spin_orbital_integrals(hcore_mo: np.ndarray, eri_mo: np.ndarray, nocc: int):
    """Return spin-orbital h and antisymmetrized g in occupied/virtual ordering."""
    nspatial = hcore_mo.shape[0]
    occ, virt = spin_modes(nspatial, nocc)
    modes = occ + virt
    nspin = len(modes)
    h = np.zeros((nspin, nspin), dtype=float)
    g = np.zeros((nspin, nspin, nspin, nspin), dtype=float)

    for p, (ps, pspin) in enumerate(modes):
        for q, (qs, qspin) in enumerate(modes):
            if pspin == qspin:
                h[p, q] = hcore_mo[ps, qs]

    for p, (ps, pspin) in enumerate(modes):
        for q, (qs, qspin) in enumerate(modes):
            for r, (rs, rspin) in enumerate(modes):
                for s, (ss, sspin) in enumerate(modes):
                    direct = (
                        eri_mo[ps, qs, rs, ss]
                        if pspin == rspin and qspin == sspin
                        else 0.0
                    )
                    exchange = (
                        eri_mo[ps, qs, ss, rs]
                        if pspin == sspin and qspin == rspin
                        else 0.0
                    )
                    g[p, q, r, s] = direct - exchange
    return h, g, modes, occ, virt


def zero_tensors(rank: int, nocc: int, nvir: int) -> dict[str, np.ndarray]:
    sizes = {"o": nocc, "v": nvir}
    tensors = {"": np.array(0.0, dtype=float)}
    for body_rank in range(1, rank + 1):
        for ann_spaces in itertools.product("ov", repeat=body_rank):
            for cre_spaces in itertools.product("ov", repeat=body_rank):
                key = "".join(ann_spaces + cre_spaces)
                tensors[key] = np.zeros(
                    tuple(sizes[space] for space in key), dtype=float
                )
    return tensors


def normal_ordered_hamiltonian_tensors(data: WickdDSRGData, rank: int):
    """Return determinant-normal-ordered Hamiltonian tensors through ``rank``."""
    h, g, modes, occ, virt = spin_orbital_integrals(
        data.hcore_mo, data.eri_mo, data.nocc
    )
    nocc = len(occ)
    nvir = len(virt)
    tensors = zero_tensors(rank, nocc, nvir)

    scalar = data.scalar
    scalar += np.einsum("ii->", h[:nocc, :nocc], optimize=True)
    scalar += 0.5 * np.einsum("ijij->", g[:nocc, :nocc, :nocc, :nocc], optimize=True)
    tensors[""] = np.array(scalar, dtype=float)

    fock = h.copy()
    fock += np.einsum("piqi->pq", g[:, :nocc, :, :nocc], optimize=True)
    ranges = {"o": np.arange(nocc), "v": np.arange(nocc, nocc + nvir)}

    for ann_space, cre_space in itertools.product("ov", repeat=2):
        key = ann_space + cre_space
        ann = ranges[ann_space]
        cre = ranges[cre_space]
        tensors[key][...] = fock[np.ix_(cre, ann)].T

    if rank >= 2:
        for ann_spaces in itertools.product("ov", repeat=2):
            for cre_spaces in itertools.product("ov", repeat=2):
                key = "".join(ann_spaces + cre_spaces)
                ann0, ann1 = (ranges[space] for space in ann_spaces)
                cre0, cre1 = (ranges[space] for space in cre_spaces)
                block = g[np.ix_(cre0, cre1, ann0, ann1)]
                tensors[key][...] = np.transpose(block, (2, 3, 0, 1))

    return tensors, modes, occ, virt


def regularized_denominator(denominator: float, flow_param: float) -> float:
    """Return the DSRG regularizer ``(1 - exp(-s Delta^2)) / Delta``."""
    if abs(denominator) < SCREEN:
        return flow_param * denominator
    return (1.0 - math.exp(-flow_param * denominator * denominator)) / denominator


def enumerate_spin_conserving_excitations(
    nspatial: int, nocc_spatial: int, eps: Sequence[float], max_rank: int
) -> tuple[dict, ...]:
    occ, virt = spin_modes(nspatial, nocc_spatial)
    occ_index = {mode: idx for idx, mode in enumerate(occ)}
    virt_index = {mode: idx for idx, mode in enumerate(virt)}
    excitations = []
    highest_rank = min(max_rank, len(occ), len(virt))
    for rank in range(1, highest_rank + 1):
        for ann in itertools.combinations(occ, rank):
            ann_spins = sorted(spin for _, spin in ann)
            for cre in itertools.combinations(virt, rank):
                if ann_spins != sorted(spin for _, spin in cre):
                    continue
                denom = sum(eps[i] for i, _ in ann) - sum(eps[a] for a, _ in cre)
                excitations.append(
                    {
                        "rank": rank,
                        "ann": ann,
                        "cre": cre,
                        "ann_idx": tuple(occ_index[mode] for mode in ann),
                        "cre_idx": tuple(virt_index[mode] for mode in cre),
                        "denom": float(denom),
                    }
                )
    return tuple(excitations)


def permutation_parity(perm: Sequence[int]) -> int:
    inversions = 0
    for i, value in enumerate(perm):
        for other in perm[i + 1 :]:
            if value > other:
                inversions += 1
    return -1 if inversions % 2 else 1


def add_antisymmetric_amplitude(
    tensor: np.ndarray,
    ann_indices: Sequence[int],
    cre_indices: Sequence[int],
    value: float,
) -> None:
    rank = len(ann_indices)
    for ann_perm in itertools.permutations(range(rank)):
        ann_sign = permutation_parity(ann_perm)
        ann_tuple = tuple(ann_indices[pos] for pos in ann_perm)
        for cre_perm in itertools.permutations(range(rank)):
            cre_sign = permutation_parity(cre_perm)
            cre_tuple = tuple(cre_indices[pos] for pos in cre_perm)
            tensor[ann_tuple + cre_tuple] += ann_sign * cre_sign * value


def amplitudes_to_tensors(
    excitations: Sequence[dict],
    amplitudes: np.ndarray,
    nocc: int,
    nvir: int,
    rank: int,
) -> dict[str, np.ndarray]:
    tensors = {}
    for excitation_rank in range(1, rank + 1):
        exc_key = "o" * excitation_rank + "v" * excitation_rank
        deexc_key = "v" * excitation_rank + "o" * excitation_rank
        shape = (nocc,) * excitation_rank + (nvir,) * excitation_rank
        tensors[exc_key] = np.zeros(shape, dtype=float)
        tensors[deexc_key] = np.zeros(
            (nvir,) * excitation_rank + (nocc,) * excitation_rank
        )

    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) <= SCREEN:
            continue
        excitation_rank = excitation["rank"]
        exc_key = "o" * excitation_rank + "v" * excitation_rank
        add_antisymmetric_amplitude(
            tensors[exc_key],
            excitation["ann_idx"],
            excitation["cre_idx"],
            float(amplitude),
        )

    for excitation_rank in range(1, rank + 1):
        exc_key = "o" * excitation_rank + "v" * excitation_rank
        deexc_key = "v" * excitation_rank + "o" * excitation_rank
        axes = tuple(range(excitation_rank, 2 * excitation_rank)) + tuple(
            range(excitation_rank)
        )
        tensors[deexc_key][...] = np.transpose(tensors[exc_key], axes)
    return tensors


def _block_shape(key: str, nocc: int, nvir: int) -> tuple[int, ...]:
    sizes = {"o": nocc, "v": nvir}
    return tuple(sizes[space] for space in key)


def _result_key_from_equation(block_key: str, equations) -> str:
    if block_key == "|":
        return ""
    for equation in equations:
        lhs = equation.compile("einsum").split("+=", 1)[0].strip()
        if lhs.startswith("y"):
            return lhs[1:]
    raise RuntimeError(f"Could not infer Wick&d result key for block {block_key}")


def _compile_block_function(block_key: str, equations, nocc: int, nvir: int):
    result_key = _result_key_from_equation(block_key, equations)
    result_var = "y" if result_key == "" else f"y{result_key}"
    code = [f"def eval_{result_var}(x, t):"]
    if result_key == "":
        code.append("    y = 0.0")
    else:
        code.append(
            f"    {result_var} = np.zeros({_block_shape(result_key, nocc, nvir)!r}, dtype=float)"
        )
    for equation in equations:
        contraction = equation.compile("einsum").replace(
            'optimize="optimal"', "optimize=True"
        )
        code.append(f"    {contraction}")
    code.append(f"    return {result_var}")
    namespace = {"np": np}
    exec("\n".join(code), namespace)
    return result_key, namespace[f"eval_{result_var}"]


@lru_cache(maxsize=8)
def make_wickd_commutator(rank: int, nocc: int, nvir: int) -> WickdCommutator:
    """Generate and compile ``[X, A]`` equations through many-body rank ``rank``."""
    _require_wickd()
    t0 = time.perf_counter()
    w.reset_space()
    w.add_space("o", "fermion", "occupied", ["i", "j", "k", "l", "m", "n", "p", "q"])
    w.add_space("v", "fermion", "unoccupied", ["a", "b", "c", "d", "e", "f", "g", "h"])

    x_op = w.op("E_0", [""])
    for body_rank in range(1, rank + 1):
        x_op = x_op + w.utils.gen_op("x", body_rank, "ov", "ov")

    t_op = None
    for body_rank in range(1, rank + 1):
        component = " ".join(["v+"] * body_rank + ["o"] * body_rank)
        term = w.op("t", [component], unique=False)
        t_op = term if t_op is None else t_op + term
    a_op = t_op - t_op.adjoint()
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    wt = w.WickTheorem()
    wt.set_single_threaded(True)
    expression = wt.contract(w.commutator(x_op, a_op), 0, 2 * rank).canonicalize()
    contract_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    many_body_equations = expression.to_manybody_equation("y")
    equation_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled = []
    for block_key, equations in sorted(many_body_equations.items()):
        compiled.append(_compile_block_function(block_key, equations, nocc, nvir))
    compile_s = time.perf_counter() - t0

    return WickdCommutator(
        rank=rank,
        functions=tuple(compiled),
        blocks=tuple(result_key for result_key, _ in compiled),
        generation_s={
            "build": build_s,
            "contract": contract_s,
            "to_manybody_equation": equation_s,
            "compile": compile_s,
            "total": build_s + contract_s + equation_s + compile_s,
        },
        n_equations=sum(len(equations) for equations in many_body_equations.values()),
        n_expression_terms=len(expression),
    )


def evaluate_commutator(
    commutator: WickdCommutator,
    x: dict[str, np.ndarray],
    t: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {key: function(x, t) for key, function in commutator.functions}


def _antisymmetrize_positions(
    tensor: np.ndarray, positions: Sequence[int]
) -> np.ndarray:
    if len(positions) <= 1:
        return tensor
    result = np.zeros_like(tensor)
    for perm in itertools.permutations(range(len(positions))):
        axes = list(range(tensor.ndim))
        for src_pos, perm_pos in zip(positions, perm):
            axes[src_pos] = positions[perm_pos]
        result += permutation_parity(perm) * np.transpose(tensor, axes)
    return result


def _antisymmetrize_block(key: str, tensor: np.ndarray) -> np.ndarray:
    if key == "" or len(key) < 4:
        return tensor
    rank = len(key) // 2
    result = tensor
    for offset, spaces in ((0, key[:rank]), (rank, key[rank:])):
        for space in "ov":
            positions = [
                offset + idx for idx, label in enumerate(spaces) if label == space
            ]
            result = _antisymmetrize_positions(result, positions)
    return result


def _complete_tensors(
    tensors: dict[str, np.ndarray], rank: int, nocc: int, nvir: int
) -> dict[str, np.ndarray]:
    completed = zero_tensors(rank, nocc, nvir)
    for key, value in tensors.items():
        if key in completed:
            completed[key][...] = _antisymmetrize_block(key, value)
    return completed


def _add_scaled_tensors(
    target: dict[str, np.ndarray], source: dict[str, np.ndarray], scale: float
) -> None:
    for key, value in source.items():
        target[key] += scale * value


def _tensor_norm(tensors: dict[str, np.ndarray]) -> float:
    return math.sqrt(
        sum(float(np.vdot(value, value).real) for value in tensors.values())
    )


def bch_hbar_wickd(
    h0: dict[str, np.ndarray],
    t: dict[str, np.ndarray],
    commutator: WickdCommutator,
    rank: int,
    nocc: int,
    nvir: int,
    max_commutators: int = 20,
    commutator_threshold: float = SCREEN,
) -> tuple[dict[str, np.ndarray], tuple[float, ...]]:
    """Evaluate the BCH expansion using the compiled Wick&d commutator."""
    hbar = {key: value.copy() for key, value in h0.items()}
    nested = {key: value.copy() for key, value in h0.items()}
    norms = []
    for ncomm in range(1, max_commutators + 1):
        nested = _complete_tensors(
            evaluate_commutator(commutator, nested, t), rank, nocc, nvir
        )
        scale = 1.0 / math.factorial(ncomm)
        _add_scaled_tensors(hbar, nested, scale)
        norm = _tensor_norm(nested) * abs(scale)
        norms.append(norm)
        if norm < commutator_threshold:
            break
    return hbar, tuple(norms)


def excitation_values(
    tensors: dict[str, np.ndarray], excitations: Sequence[dict]
) -> np.ndarray:
    values = np.zeros(len(excitations), dtype=float)
    for idx, excitation in enumerate(excitations):
        rank = excitation["rank"]
        key = "o" * rank + "v" * rank
        values[idx] = tensors[key][excitation["ann_idx"] + excitation["cre_idx"]]
    return values


def solve_wickd_dsrg(
    data: WickdDSRGData,
    max_rank: int = 2,
    flow_param: float = 5.0,
    e_tol: float = 1.0e-10,
    r_tol: float = 1.0e-5,
    maxiter: int = 80,
    max_commutators: int = 20,
    do_diis: bool = True,
    diis_start: int = 3,
    diis_nvec: int = 8,
    commutator_threshold: float = SCREEN,
) -> WickdDSRGResult:
    """Run single-reference DSRG(n) using Wick&d-generated equations."""
    if max_rank < 1:
        raise ValueError("max_rank must be positive")

    h0, _, occ, virt = normal_ordered_hamiltonian_tensors(data, max_rank)
    nocc = len(occ)
    nvir = len(virt)
    excitations = list(
        enumerate_spin_conserving_excitations(data.nmo, data.nocc, data.eps, max_rank)
    )
    for excitation in excitations:
        excitation["regularized_denominator"] = regularized_denominator(
            excitation["denom"], flow_param
        )

    commutator = make_wickd_commutator(max_rank, nocc, nvir)
    h0_offdiag = excitation_values(h0, excitations)
    amplitudes = np.array(
        [
            h0_offdiag[idx] * excitation["regularized_denominator"]
            for idx, excitation in enumerate(excitations)
        ],
        dtype=float,
    )
    diis = DIIS(
        diis_start=diis_start,
        diis_nvec=diis_nvec,
        diis_min=min(diis_start, diis_nvec),
        do_diis=do_diis,
    )

    previous_energy = None
    history = []
    start_s = time.perf_counter()
    hbar = h0
    converged = False

    for iteration in range(maxiter + 1):
        iter_s = time.perf_counter()
        t_tensors = amplitudes_to_tensors(excitations, amplitudes, nocc, nvir, max_rank)
        hbar, commutator_norms = bch_hbar_wickd(
            h0,
            t_tensors,
            commutator,
            max_rank,
            nocc,
            nvir,
            max_commutators=max_commutators,
            commutator_threshold=commutator_threshold,
        )
        energy = float(hbar[""])
        hbar_offdiag = excitation_values(hbar, excitations)
        fixed_point = np.array(
            [
                (hbar_offdiag[idx] + excitation["denom"] * amplitudes[idx])
                * excitation["regularized_denominator"]
                for idx, excitation in enumerate(excitations)
            ],
            dtype=float,
        )
        update = fixed_point - amplitudes
        rms_update = float(np.linalg.norm(update))
        delta_energy = 0.0 if previous_energy is None else energy - previous_energy
        history.append(
            WickdDSRGIteration(
                iteration=iteration,
                energy=energy,
                delta_energy=delta_energy,
                rms_update=rms_update,
                ncomm=len(commutator_norms),
                iter_s=time.perf_counter() - iter_s,
            )
        )

        if (
            previous_energy is not None
            and abs(delta_energy) < e_tol
            and rms_update < r_tol
        ):
            converged = True
            break

        amplitudes = diis.update(fixed_point, update)
        previous_energy = energy

    return WickdDSRGResult(
        energy=energy,
        converged=converged,
        iterations=len(history),
        max_rank=max_rank,
        amplitudes=amplitudes,
        history=tuple(history),
        hbar=hbar,
        equations=commutator,
        seconds=time.perf_counter() - start_s,
    )


def _solve_fixed_rank(rank: int, data: WickdDSRGData, **kwargs) -> WickdDSRGResult:
    requested_rank = kwargs.pop("max_rank", rank)
    if requested_rank != rank:
        raise ValueError(f"solve_wickd_dsrg{rank} requires max_rank={rank}")
    return solve_wickd_dsrg(data, max_rank=rank, **kwargs)


def solve_wickd_dsrg2(data: WickdDSRGData, **kwargs) -> WickdDSRGResult:
    """Run single-reference DSRG(2) using Wick&d-generated equations."""
    return _solve_fixed_rank(2, data, **kwargs)


def solve_wickd_dsrg3(data: WickdDSRGData, **kwargs) -> WickdDSRGResult:
    """Run single-reference DSRG(3) using Wick&d-generated equations."""
    return _solve_fixed_rank(3, data, **kwargs)


def solve_wickd_dsrg4(data: WickdDSRGData, **kwargs) -> WickdDSRGResult:
    """Run single-reference DSRG(4) using Wick&d-generated equations."""
    return _solve_fixed_rank(4, data, **kwargs)


WickdDSRG = solve_wickd_dsrg
