import argparse
import contextlib
import io
import itertools
import json
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wickd as w

from forte2 import CI, Determinant, RHF, State, System
from forte2.base_classes import CIParams, DavidsonLiuParams
from forte2.helpers import DIIS, logger

SCREEN = 1.0e-12
BASIS = "sto-3g"
DEFAULT_FLOW_PARAM = 5.0
DEFAULT_E_TOL = 1.0e-10
DEFAULT_R_TOL = 1.0e-5
DEFAULT_MAX_ITER = 80
DEFAULT_MAX_COMM = 20

REFERENCE_DSRG = {
    2: {
        2: -1.137831710258920,
        3: -1.137283834651968,
        4: -1.137283834651968,
    },
    4: {
        2: -2.140264303595860,
        3: -2.138895129858825,
        4: -2.138889864332239,
    },
    6: {
        2: -3.144610080816127,
        3: -3.142386244444168,
        4: -3.142365107876118,
    },
}


@dataclass(frozen=True)
class MoleculeData:
    natoms: int
    spacing: float
    system: System
    rhf: RHF
    hcore_mo: np.ndarray
    eri_mo: np.ndarray
    eps: np.ndarray
    reference: Determinant


@dataclass(frozen=True)
class WickdCommutator:
    rank: int
    functions: tuple
    blocks: tuple[str, ...]
    generation_s: dict[str, float]
    n_equations: int
    n_expression_terms: int


def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def determinant_count(nmo, na, nb):
    return math.comb(nmo, na) * math.comb(nmo, nb)


def build_linear_h_data(natoms, spacing):
    xyz = "\n".join(f"H 0.0 0.0 {i * spacing:.12f}" for i in range(natoms))
    with contextlib.redirect_stdout(io.StringIO()):
        system = System(
            xyz=xyz,
            basis_set=BASIS,
            minao_basis_set=None,
            cholesky_tei=True,
            cholesky_tol=1.0e-12,
        )
        rhf = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system)
        rhf.run()

    coeff = rhf.C[0]
    hcore_mo = np.einsum(
        "pq,pi,qj->ij", system.ints_hcore(), coeff, coeff, optimize=True
    )
    eri_mo = system.fock_builder.two_electron_integrals_block(coeff)
    reference = Determinant("2" * rhf.na + "0" * (rhf.nmo - rhf.na))
    return MoleculeData(
        natoms=natoms,
        spacing=spacing,
        system=system,
        rhf=rhf,
        hcore_mo=hcore_mo,
        eri_mo=eri_mo,
        eps=np.array(rhf.eps[0]),
        reference=reference,
    )


def forte2_fci_energy(data):
    ndet = determinant_count(data.rhf.nmo, data.rhf.na, data.rhf.nb)
    algorithm = "exact" if ndet <= 10000 else "hz"
    params = CIParams(ci_algorithm=algorithm)
    davidson = DavidsonLiuParams(e_tol=1.0e-10, r_tol=1.0e-5, maxiter=200)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        ci = CI(
            states=State(nel=data.natoms, multiplicity=1, ms=0.0),
            active_orbitals=list(range(data.rhf.nmo)),
            ci_params=params,
            davidson_liu_params=davidson,
        )(data.rhf)
        ci.run()
    return {
        "energy": float(ci.E[0]),
        "seconds": time.perf_counter() - t0,
        "algorithm": algorithm,
        "converged": bool(ci.get_convergence_status()[0]),
        "ndet": ndet,
    }


def spin_modes(nspatial, nocc):
    occ = [(i, spin) for i in range(nocc) for spin in ("a", "b")]
    virt = [(a, spin) for a in range(nocc, nspatial) for spin in ("a", "b")]
    return occ, virt


def spin_orbital_integrals(hcore_mo, eri_mo, nocc):
    nspatial = hcore_mo.shape[0]
    occ, virt = spin_modes(nspatial, nocc)
    modes = occ + virt
    nspin = len(modes)
    h = np.zeros((nspin, nspin))
    g = np.zeros((nspin, nspin, nspin, nspin))

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


def normal_ordered_hamiltonian_tensors(data, rank):
    h, g, modes, occ, virt = spin_orbital_integrals(
        data.hcore_mo, data.eri_mo, data.rhf.na
    )
    nocc = len(occ)
    nvir = len(virt)
    occupied = np.arange(nocc)

    scalar = data.system.nuclear_repulsion
    scalar += np.einsum("ii->", h[:nocc, :nocc], optimize=True)
    scalar += 0.5 * np.einsum("ijij->", g[:nocc, :nocc, :nocc, :nocc], optimize=True)

    fock = h.copy()
    fock += np.einsum("piqi->pq", g[:, :nocc, :, :nocc], optimize=True)

    tensors = {"": np.array(scalar, dtype=float)}
    sizes = {"o": nocc, "v": nvir}

    for body_rank in range(1, rank + 1):
        for ann_spaces in itertools.product("ov", repeat=body_rank):
            for cre_spaces in itertools.product("ov", repeat=body_rank):
                key = "".join(ann_spaces + cre_spaces)
                shape = tuple(sizes[space] for space in key)
                tensors[key] = np.zeros(shape, dtype=float)

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

    return tensors, modes, occ, virt, occupied


def permutation_parity(perm):
    inversions = 0
    for i, value in enumerate(perm):
        for other in perm[i + 1 :]:
            if value > other:
                inversions += 1
    return -1 if inversions % 2 else 1


def regularized_denominator(denom, flow_param):
    return (1.0 - math.exp(-flow_param * denom * denom)) / denom


def enumerate_spin_conserving_excitations(
    nspatial, nocc_spatial, eps, max_excitation_rank
):
    occ, virt = spin_modes(nspatial, nocc_spatial)
    occ_index = {mode: idx for idx, mode in enumerate(occ)}
    virt_index = {mode: idx for idx, mode in enumerate(virt)}
    highest_rank = min(max_excitation_rank, len(occ), len(virt))
    excitations = []

    for rank in range(1, highest_rank + 1):
        for ann in itertools.combinations(occ, rank):
            ann_spins = sorted(spin for _, spin in ann)
            for cre in itertools.combinations(virt, rank):
                if ann_spins != sorted(spin for _, spin in cre):
                    continue
                ann_idx = tuple(occ_index[mode] for mode in ann)
                cre_idx = tuple(virt_index[mode] for mode in cre)
                denom = sum(eps[i] for i, _ in ann) - sum(eps[a] for a, _ in cre)
                excitations.append(
                    {
                        "rank": rank,
                        "ann": ann,
                        "cre": cre,
                        "ann_idx": ann_idx,
                        "cre_idx": cre_idx,
                        "denom": denom,
                        "regularized_denominator": regularized_denominator(
                            denom, DEFAULT_FLOW_PARAM
                        ),
                    }
                )
    return excitations


def add_antisymmetric_amplitude(tensor, ann_indices, cre_indices, value):
    rank = len(ann_indices)
    for ann_perm in itertools.permutations(range(rank)):
        ann_sign = permutation_parity(ann_perm)
        ann_tuple = tuple(ann_indices[pos] for pos in ann_perm)
        for cre_perm in itertools.permutations(range(rank)):
            cre_sign = permutation_parity(cre_perm)
            cre_tuple = tuple(cre_indices[pos] for pos in cre_perm)
            tensor[ann_tuple + cre_tuple] += ann_sign * cre_sign * value


def amplitudes_to_tensors(excitations, amplitudes, nocc, nvir, rank):
    sizes = {"o": nocc, "v": nvir}
    tensors = {}
    for excitation_rank in range(1, rank + 1):
        exc_key = "o" * excitation_rank + "v" * excitation_rank
        deexc_key = "v" * excitation_rank + "o" * excitation_rank
        tensors[exc_key] = np.zeros(
            tuple(sizes[space] for space in exc_key), dtype=float
        )
        tensors[deexc_key] = np.zeros(
            tuple(sizes[space] for space in deexc_key), dtype=float
        )

    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) <= SCREEN:
            continue
        excitation_rank = excitation["rank"]
        exc_key = "o" * excitation_rank + "v" * excitation_rank
        add_antisymmetric_amplitude(
            tensors[exc_key], excitation["ann_idx"], excitation["cre_idx"], amplitude
        )

    for excitation_rank in range(1, rank + 1):
        exc_key = "o" * excitation_rank + "v" * excitation_rank
        deexc_key = "v" * excitation_rank + "o" * excitation_rank
        axes = tuple(range(excitation_rank, 2 * excitation_rank)) + tuple(
            range(excitation_rank)
        )
        tensors[deexc_key][...] = np.transpose(tensors[exc_key], axes)

    return tensors


def block_shape_from_tensor_key(key, nocc, nvir):
    sizes = {"o": nocc, "v": nvir}
    return tuple(sizes[space] for space in key)


def result_key_from_equation(block_key, equations):
    if block_key == "|":
        return ""
    for equation in equations:
        lhs = equation.compile("einsum").split("+=", 1)[0].strip()
        if lhs.startswith("y"):
            return lhs[1:]
    raise RuntimeError(f"Could not infer result key for block {block_key}")


def compile_block_function(block_key, equations, nocc, nvir):
    result_key = result_key_from_equation(block_key, equations)
    result_var = "y" if result_key == "" else f"y{result_key}"
    code = [f"def eval_{result_var}(x, t):"]
    if result_key == "":
        code.append("    y = 0.0")
    else:
        shape = block_shape_from_tensor_key(result_key, nocc, nvir)
        code.append(f"    {result_var} = np.zeros({shape!r}, dtype=float)")
    for equation in equations:
        contraction = equation.compile("einsum").replace(
            'optimize="optimal"', "optimize=True"
        )
        code.append(f"    {contraction}")
    code.append(f"    return {result_var}")
    namespace = {"np": np}
    exec("\n".join(code), namespace)
    return result_key, namespace[f"eval_{result_var}"]


def make_wickd_commutator(rank, nocc, nvir):
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
        result_key, function = compile_block_function(block_key, equations, nocc, nvir)
        compiled.append((result_key, function))
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


def zero_x_tensors(rank, nocc, nvir):
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


def evaluate_commutator(commutator, x, t):
    y = {}
    for key, function in commutator.functions:
        y[key] = function(x, t)
    return y


def antisymmetrize_positions(tensor, positions):
    if len(positions) <= 1:
        return tensor

    result = np.zeros_like(tensor)
    for perm in itertools.permutations(range(len(positions))):
        axes = list(range(tensor.ndim))
        for src_pos, perm_pos in zip(positions, perm):
            axes[src_pos] = positions[perm_pos]
        result += permutation_parity(perm) * np.transpose(tensor, axes)
    return result


def antisymmetrize_block(key, tensor):
    if key == "" or len(key) < 4:
        return tensor

    rank = len(key) // 2
    result = tensor
    for offset, spaces in ((0, key[:rank]), (rank, key[rank:])):
        for space in "ov":
            positions = [
                offset + idx for idx, label in enumerate(spaces) if label == space
            ]
            result = antisymmetrize_positions(result, positions)
    return result


def antisymmetrize_tensors(tensors):
    return {key: antisymmetrize_block(key, value) for key, value in tensors.items()}


def add_scaled_tensors(target, source, scale):
    for key, value in source.items():
        if key not in target:
            target[key] = scale * value.copy()
        else:
            target[key] += scale * value


def tensor_norm(tensors):
    total = 0.0
    for value in tensors.values():
        total += float(np.vdot(value, value).real)
    return math.sqrt(total)


def truncate_missing_blocks(tensors, rank, nocc, nvir):
    completed = zero_x_tensors(rank, nocc, nvir)
    for key, value in tensors.items():
        if key in completed:
            completed[key][...] = value
    return completed


def bch_hbar_wickd(h0, t, commutator, rank, nocc, nvir, max_comm=DEFAULT_MAX_COMM):
    hbar = {key: value.copy() for key, value in h0.items()}
    nested = {key: value.copy() for key, value in h0.items()}
    commutator_norms = []

    for ncomm in range(1, max_comm + 1):
        nested = truncate_missing_blocks(
            antisymmetrize_tensors(evaluate_commutator(commutator, nested, t)),
            rank,
            nocc,
            nvir,
        )
        scale = 1.0 / math.factorial(ncomm)
        add_scaled_tensors(hbar, nested, scale)
        norm = tensor_norm(nested) * abs(scale)
        commutator_norms.append(norm)
        if norm < SCREEN:
            break
    return hbar, commutator_norms


def excitation_values(tensors, excitations):
    values = np.zeros(len(excitations), dtype=float)
    for idx, excitation in enumerate(excitations):
        rank = excitation["rank"]
        key = "o" * rank + "v" * rank
        values[idx] = tensors[key][excitation["ann_idx"] + excitation["cre_idx"]]
    return values


def solve_wickd_dsrg(
    data,
    rank,
    flow_param=DEFAULT_FLOW_PARAM,
    e_tol=DEFAULT_E_TOL,
    r_tol=DEFAULT_R_TOL,
    max_iter=DEFAULT_MAX_ITER,
):
    h0, modes, occ, virt, _ = normal_ordered_hamiltonian_tensors(data, rank)
    nocc = len(occ)
    nvir = len(virt)
    excitations = enumerate_spin_conserving_excitations(
        data.rhf.nmo, data.rhf.na, data.eps, rank
    )
    for excitation in excitations:
        excitation["regularized_denominator"] = regularized_denominator(
            excitation["denom"], flow_param
        )

    commutator = make_wickd_commutator(rank, nocc, nvir)
    h0_offdiag = excitation_values(h0, excitations)
    amplitudes = np.array(
        [
            h0_offdiag[idx] * excitation["regularized_denominator"]
            for idx, excitation in enumerate(excitations)
        ],
        dtype=float,
    )
    diis = DIIS(diis_start=3, diis_nvec=8, diis_min=3, do_diis=True)
    history = []
    previous_energy = None
    t0 = time.perf_counter()

    for iteration in range(max_iter + 1):
        iter_t0 = time.perf_counter()
        t_tensors = amplitudes_to_tensors(excitations, amplitudes, nocc, nvir, rank)
        hbar, commutator_norms = bch_hbar_wickd(
            h0, t_tensors, commutator, rank, nocc, nvir
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
            {
                "iteration": iteration,
                "energy": energy,
                "delta_energy": delta_energy,
                "rms_update": rms_update,
                "ncomm": len(commutator_norms),
                "iter_s": time.perf_counter() - iter_t0,
                "diis_status": getattr(diis, "status", ""),
            }
        )

        if (
            previous_energy is not None
            and abs(delta_energy) < e_tol
            and rms_update < r_tol
        ):
            return {
                "status": "ok",
                "energy": energy,
                "iterations": iteration + 1,
                "n_amplitudes": len(excitations),
                "solve_s": time.perf_counter() - t0,
                "history_tail": history[-5:],
                "equations": {
                    "generation_s": commutator.generation_s,
                    "n_equations": commutator.n_equations,
                    "n_expression_terms": commutator.n_expression_terms,
                    "n_blocks": len(commutator.blocks),
                },
            }

        amplitudes = diis.update(fixed_point, update)
        previous_energy = energy

    return {
        "status": "not_converged",
        "energy": float(history[-1]["energy"]),
        "iterations": len(history),
        "n_amplitudes": len(excitations),
        "solve_s": time.perf_counter() - t0,
        "history_tail": history[-5:],
        "equations": {
            "generation_s": commutator.generation_s,
            "n_equations": commutator.n_equations,
            "n_expression_terms": commutator.n_expression_terms,
            "n_blocks": len(commutator.blocks),
        },
    }


def load_sparse_reference(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    lookup = {}
    for row in data.get("cases", []):
        if row.get("status") != "ok":
            continue
        key = (row.get("natoms"), row.get("spacing"), row.get("rank"))
        lookup[key] = row
    return lookup


def run_case(natoms, spacing, rank, sparse_lookup):
    logger.set_verbosity_level(0)
    t0 = time.perf_counter()
    data = build_linear_h_data(natoms, spacing)
    setup_s = time.perf_counter() - t0

    fci = forte2_fci_energy(data)
    result = solve_wickd_dsrg(data, rank)
    reference_energy = (
        REFERENCE_DSRG.get(natoms, {}).get(rank)
        if abs(spacing - 0.74) < 1.0e-12
        else None
    )
    sparse_row = sparse_lookup.get((natoms, spacing, rank), {})

    return {
        "status": result["status"],
        "natoms": natoms,
        "spacing": spacing,
        "rank": rank,
        "basis": BASIS,
        "flow_param": DEFAULT_FLOW_PARAM,
        "e_tol": DEFAULT_E_TOL,
        "r_tol": DEFAULT_R_TOL,
        "nmo": data.rhf.nmo,
        "na": data.rhf.na,
        "nb": data.rhf.nb,
        "reference": data.reference.str(data.rhf.nmo),
        "rhf_energy": float(data.rhf.E),
        "setup_s": setup_s,
        "fci": fci,
        "wickd_energy": result["energy"],
        "reference_sparse_energy": reference_energy,
        "wickd_minus_reference_sparse": (
            None if reference_energy is None else result["energy"] - reference_energy
        ),
        "wickd_minus_fci": result["energy"] - fci["energy"],
        "wickd_solve_s": result["solve_s"],
        "wickd_iterations": result["iterations"],
        "n_amplitudes": result["n_amplitudes"],
        "wickd_history_tail": result["history_tail"],
        "wickd_equations": result["equations"],
        "sparse_reference": {
            "energy": sparse_row.get("dsrg_energy", reference_energy),
            "solve_s": sparse_row.get("solve_s"),
            "iterations": sparse_row.get("iterations"),
        },
        "completed_at": now_s(),
    }


def run_case_with_timeout(natoms, spacing, rank, sparse_lookup, timeout_s):
    if timeout_s is None or timeout_s <= 0.0:
        return run_case(natoms, spacing, rank, sparse_lookup)

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(
        target=lambda q: q.put(run_case(natoms, spacing, rank, sparse_lookup)),
        args=(queue,),
    )
    t0 = time.perf_counter()
    process.start()
    process.join(timeout_s)
    elapsed_s = time.perf_counter() - t0
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "status": "timeout",
            "natoms": natoms,
            "spacing": spacing,
            "rank": rank,
            "basis": BASIS,
            "flow_param": DEFAULT_FLOW_PARAM,
            "timeout_s": timeout_s,
            "elapsed_s": elapsed_s,
            "completed_at": now_s(),
        }
    if queue.empty():
        return {
            "status": "no-result",
            "natoms": natoms,
            "spacing": spacing,
            "rank": rank,
            "basis": BASIS,
            "flow_param": DEFAULT_FLOW_PARAM,
            "elapsed_s": elapsed_s,
            "exitcode": process.exitcode,
            "completed_at": now_s(),
        }
    row = queue.get()
    row["wall_s"] = elapsed_s
    return row


def save_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="experiments/wickd_dsrg_equations_results.json"
    )
    parser.add_argument(
        "--sparse-results", default="experiments/dsrg_hchain_results.json"
    )
    parser.add_argument("--atoms", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--spacing", type=float, default=0.74)
    parser.add_argument("--case-timeout", type=float, default=300.0)
    args = parser.parse_args()

    sparse_lookup = load_sparse_reference(Path(args.sparse_results))
    payload = {
        "metadata": {
            "created_at": now_s(),
            "basis": BASIS,
            "screen": SCREEN,
            "flow_param": DEFAULT_FLOW_PARAM,
            "e_tol": DEFAULT_E_TOL,
            "r_tol": DEFAULT_R_TOL,
            "max_iter": DEFAULT_MAX_ITER,
            "max_comm": DEFAULT_MAX_COMM,
            "method": "Wick&d-generated spin-orbital single-commutator BCH DSRG(n)",
        },
        "cases": [],
    }
    save_results(Path(args.output), payload)

    for natoms in args.atoms:
        for rank in args.ranks:
            print(f"START {now_s()} H{natoms} DSRG({rank}) Wick&d", flush=True)
            row = run_case_with_timeout(
                natoms, args.spacing, rank, sparse_lookup, args.case_timeout
            )
            payload["cases"].append(row)
            save_results(Path(args.output), payload)
            diff = row.get("wickd_minus_reference_sparse")
            diff_text = "n/a" if diff is None else f"{diff:.3e}"
            energy = row.get("wickd_energy")
            energy_text = "n/a" if energy is None else f"{energy:.15f}"
            print(
                f"DONE H{natoms} DSRG({rank}) status={row['status']} "
                f"E={energy_text} diff_sparse={diff_text} "
                f"iters={row.get('wickd_iterations', 'n/a')} "
                f"solve={row.get('wickd_solve_s', row.get('elapsed_s', 0.0)):.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
