import argparse
import contextlib
import io
import itertools
import json
import math
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wickd as w

from forte2.helpers import DIIS, logger

try:
    from experiments.wickd_dsrg_equations_benchmark import (
        BASIS,
        DEFAULT_E_TOL,
        DEFAULT_FLOW_PARAM,
        DEFAULT_MAX_COMM,
        DEFAULT_MAX_ITER,
        DEFAULT_R_TOL,
        REFERENCE_DSRG,
        SCREEN,
        build_linear_h_data,
        enumerate_spin_conserving_excitations,
        forte2_fci_energy,
        normal_ordered_hamiltonian_tensors,
        regularized_denominator,
    )
except ModuleNotFoundError:
    from wickd_dsrg_equations_benchmark import (
        BASIS,
        DEFAULT_E_TOL,
        DEFAULT_FLOW_PARAM,
        DEFAULT_MAX_COMM,
        DEFAULT_MAX_ITER,
        DEFAULT_R_TOL,
        REFERENCE_DSRG,
        SCREEN,
        build_linear_h_data,
        enumerate_spin_conserving_excitations,
        forte2_fci_energy,
        normal_ordered_hamiltonian_tensors,
        regularized_denominator,
    )


SPACES = "oOvV"
SPACE_KIND = {"o": "occupied", "O": "occupied", "v": "unoccupied", "V": "unoccupied"}
SPACE_INDICES = {
    "o": ["i", "j", "k", "l", "m", "n", "p", "q"],
    "O": ["I", "J", "K", "L", "M", "N", "P", "Q"],
    "v": ["a", "b", "c", "d", "e", "f", "g", "h"],
    "V": ["A", "B", "C", "D", "E", "F", "G", "H"],
}


@dataclass(frozen=True)
class SpinIntegratedCommutator:
    rank: int
    functions: tuple
    blocks: tuple[str, ...]
    generation_s: dict[str, float]
    n_equations: int
    n_expression_terms: int


@dataclass(frozen=True)
class SpinUniqueCommutator:
    rank: int
    functions: tuple
    blocks: tuple[str, ...]
    generation_s: dict[str, float]
    n_equations: int
    n_expression_terms: int
    n_unique_equations: int


def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def permutation_parity(perm):
    inversions = 0
    for i, value in enumerate(perm):
        for other in perm[i + 1 :]:
            if value > other:
                inversions += 1
    return -1 if inversions % 2 else 1


def tensor_shape(key, nocc, nvir):
    sizes = {"o": nocc, "O": nocc, "v": nvir, "V": nvir}
    return tuple(sizes[space] for space in key)


def result_key_from_equation(block_key, equations):
    if block_key == "|":
        return ""
    for equation in equations:
        for statement in compiled_einsum_statements(equation):
            lhs = statement.split("+=", 1)[0].strip()
            if lhs.startswith("y"):
                return lhs[1:]
    raise RuntimeError(f"Could not infer result key for block {block_key}")


def compiled_einsum_statements(equation):
    compiled = equation.compile("einsum")
    return tuple(
        statement.strip()
        for line in compiled.splitlines()
        for statement in line.split(";")
        if statement.strip()
    )


def compile_block_function(block_key, equations, nocc, nvir):
    result_key = result_key_from_equation(block_key, equations)
    result_var = "y" if result_key == "" else f"y{result_key}"
    code = [f"def eval_{result_var}(x, t):"]
    if result_key == "":
        code.append("    y = 0.0")
    else:
        code.append(
            f"    {result_var} = np.zeros({tensor_shape(result_key, nocc, nvir)!r})"
        )
    for equation in equations:
        code.extend(
            f"    {statement.replace('optimize=\"optimal\"', 'optimize=False')}"
            for statement in compiled_einsum_statements(equation)
        )
    code.append(f"    return {result_var}")
    namespace = {"np": np}
    exec("\n".join(code), namespace)
    return result_key, namespace[f"eval_{result_var}"]


EINSUM_RE = re.compile(
    r'(\w+)\s*\+=\s*([-+]?\d+\.\d+)\s*\*\s*np\.einsum\("([^"]+)"\s*,\s*'
    r'(.*?)\s*,\s*optimize="optimal"\)'
)
TENSOR_REF_RE = re.compile(r'(\w+)\["([^"]*)"\]')


def parse_einsum_line(line):
    match = EINSUM_RE.match(line.strip())
    if match is None:
        raise ValueError(f"Could not parse Wick&d einsum line: {line}")
    lhs, coefficient, subscripts, refs = match.groups()
    return lhs, float(coefficient), subscripts, TENSOR_REF_RE.findall(refs)


def normalize_subscripts(input_indices, output_indices):
    pool = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    mapping = {}

    def mapped(label):
        if label not in mapping:
            mapping[label] = next(pool)
        return mapping[label]

    normalized_inputs = [
        "".join(mapped(label) for label in group) for group in input_indices
    ]
    normalized_output = "".join(mapped(label) for label in output_indices)
    return ",".join(normalized_inputs) + "->" + normalized_output


def spin_count_label(rank, nalpha):
    if rank == 1:
        return "1"
    if rank == 2:
        return "aa" if nalpha == 2 else "ab"
    return f"r{rank}s{nalpha}"


def spin_count_from_label(label, rank):
    if label == "1":
        return 1
    if label == "aa":
        return 2
    if label == "ab":
        return 1
    prefix = f"r{rank}s"
    if label.startswith(prefix):
        return int(label[len(prefix) :])
    raise ValueError(f"Cannot infer spin count from label {label}")


def unique_spin_counts(rank):
    if rank == 1:
        return [1]
    return list(range((rank + 1) // 2, rank + 1))


def spin_unique_output_key(result_key):
    if result_key == "":
        return ""
    mapped = spin_unique_tensor_ref("y", result_key, result_key)
    if mapped is None:
        return None
    _, unique_name, unique_key, _ = mapped
    return f"{unique_name.split(':', 1)[1]}:{unique_key}"


def spin_unique_selected_result_keys(rank):
    selected = {""}
    for body_rank in range(1, rank + 1):
        for nalpha in unique_spin_counts(body_rank):
            for base_key in (
                "".join(item) for item in itertools.product("ov", repeat=2 * body_rank)
            ):
                ann = base_key[:body_rank]
                cre = base_key[body_rank:]
                selected_key = (
                    ann[:nalpha]
                    + ann[nalpha:].upper()
                    + cre[:nalpha]
                    + cre[nalpha:].upper()
                )
                selected.add(selected_key)
    return selected


def spin_unique_tensor_ref(name, key, indices):
    if key == "":
        return 1.0, f"{name}:", "", indices

    rank = len(key) // 2
    if rank == 1:
        if is_alpha_space(key[0]) != is_alpha_space(key[1]):
            return None
        return 1.0, f"{name}:1", key.lower(), indices

    ann_positions = list(range(rank))
    cre_positions = list(range(rank, 2 * rank))
    nalpha_ann = sum(is_alpha_space(key[pos]) for pos in ann_positions)
    nalpha_cre = sum(is_alpha_space(key[pos]) for pos in cre_positions)
    if nalpha_ann != nalpha_cre:
        return None

    flip_spin = nalpha_ann < (rank - nalpha_ann)

    def effective_alpha(pos):
        return is_alpha_space(key[pos]) != flip_spin

    unique_nalpha = max(nalpha_ann, rank - nalpha_ann)
    ann_order = [pos for pos in ann_positions if effective_alpha(pos)] + [
        pos for pos in ann_positions if not effective_alpha(pos)
    ]
    cre_order = [pos for pos in cre_positions if effective_alpha(pos)] + [
        pos for pos in cre_positions if not effective_alpha(pos)
    ]
    order = ann_order + cre_order
    sign = permutation_parity(ann_order) * permutation_parity(
        [pos - rank for pos in cre_order]
    )
    label = spin_count_label(rank, unique_nalpha)
    return (
        float(sign),
        f"{name}:{label}",
        "".join(key[pos].lower() for pos in order),
        "".join(indices[pos] for pos in order),
    )


def safe_function_suffix(key):
    if key == "":
        return "scalar"
    return key.replace(":", "_")


def compile_spin_unique_function(unique_key, terms, nocc, nvir):
    result_var = f"y_{safe_function_suffix(unique_key)}"
    code = [f"def eval_{safe_function_suffix(unique_key)}(x, t):"]
    if unique_key == "":
        code.append("    y_scalar = 0.0")
    else:
        _, base_key = unique_key.split(":", 1)
        code.append(
            f"    {result_var} = np.zeros({tensor_shape(base_key, nocc, nvir)!r})"
        )

    grouped = {}
    for coefficient, subscripts, refs in terms:
        grouped[(subscripts, tuple(refs))] = (
            grouped.get((subscripts, tuple(refs)), 0.0) + coefficient
        )

    n_unique = 0
    for (subscripts, refs), coefficient in grouped.items():
        if abs(coefficient) <= SCREEN:
            continue
        n_unique += 1
        ref_text = ",".join(spin_unique_ref_expression(name, key) for name, key in refs)
        code.append(
            f"    {result_var} += {coefficient:.12g} * "
            f'np.einsum("{subscripts}",{ref_text},optimize=False)'
        )
    code.append(f"    return {result_var}")
    namespace = {"np": np}
    exec("\n".join(code), namespace)
    return namespace[f"eval_{safe_function_suffix(unique_key)}"], n_unique


def spin_unique_ref_expression(name, key):
    tensor_name, label = name.split(":", 1)
    unique_key = "" if key == "" else f"{label}:{key}"
    return f'{tensor_name}["{unique_key}"]'


def make_spin_unique_commutator(rank, nocc, nvir):
    t0 = time.perf_counter()
    w.reset_space()
    for space in SPACES:
        w.add_space(space, "fermion", SPACE_KIND[space], SPACE_INDICES[space])

    x_op = spin_conserving_x_operator(rank)
    t_op = None
    for body_rank in range(1, rank + 1):
        for nalpha in range(body_rank + 1):
            term = w.op("t", [excitation_component(body_rank, nalpha)], unique=False)
            t_op = term if t_op is None else t_op + term
    a_op = t_op - t_op.adjoint()
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    commutator_expr = w.commutator(x_op, a_op)
    commutator_build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    wt = w.WickTheorem()
    wt.set_single_threaded(True)
    expression = wt.contract(commutator_expr, 0, 2 * rank).canonicalize()
    contract_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    many_body_equations = expression.to_manybody_equation("y")
    equation_s = time.perf_counter() - t0

    selected = spin_unique_selected_result_keys(rank)
    representative_by_output = {}
    for block_key, equations in many_body_equations.items():
        result_key = result_key_from_equation(block_key, equations)
        output_mapping = spin_unique_tensor_ref("y", result_key, result_key)
        if output_mapping is None:
            continue
        _, output_name, output_key, _ = output_mapping
        unique_output = (
            "" if output_key == "" else f"{output_name.split(':', 1)[1]}:{output_key}"
        )
        representative = representative_by_output.get(unique_output)
        if (
            representative is None
            or (result_key in selected and representative not in selected)
            or (
                (result_key in selected) == (representative in selected)
                and result_key < representative
            )
        ):
            representative_by_output[unique_output] = result_key

    terms_by_output = {}
    n_equations = 0
    n_mapped_equations = 0

    for block_key, equations in many_body_equations.items():
        result_key = result_key_from_equation(block_key, equations)
        for equation in equations:
            for statement in compiled_einsum_statements(equation):
                n_equations += 1
                lhs, coefficient, subscripts, refs = parse_einsum_line(statement)
                input_subscripts, output_subscripts = subscripts.split("->", 1)
                input_indices = input_subscripts.split(",") if input_subscripts else []
                output_mapping = spin_unique_tensor_ref(
                    "y", result_key, output_subscripts
                )
                if output_mapping is None:
                    continue
                output_sign, output_name, output_key, output_indices = output_mapping
                unique_output = (
                    ""
                    if output_key == ""
                    else f"{output_name.split(':', 1)[1]}:{output_key}"
                )
                if representative_by_output.get(unique_output) != result_key:
                    continue
                mapped_refs = []
                mapped_indices = []
                mapped_coefficient = output_sign * coefficient
                keep_term = True
                for (name, ref_key), ref_indices in zip(refs, input_indices):
                    mapped = spin_unique_tensor_ref(name, ref_key, ref_indices)
                    if mapped is None:
                        keep_term = False
                        break
                    sign, unique_name, unique_key, unique_indices = mapped
                    mapped_coefficient *= sign
                    mapped_refs.append((unique_name, unique_key))
                    mapped_indices.append(unique_indices)
                if not keep_term:
                    continue
                n_mapped_equations += 1
                normalized_subscripts = normalize_subscripts(
                    mapped_indices, output_indices
                )
                terms_by_output.setdefault(unique_output, []).append(
                    (mapped_coefficient, normalized_subscripts, tuple(mapped_refs))
                )

    t0 = time.perf_counter()
    compiled = []
    n_unique_equations = 0
    for unique_key, terms in sorted(terms_by_output.items()):
        function, n_unique = compile_spin_unique_function(unique_key, terms, nocc, nvir)
        compiled.append((unique_key, function))
        n_unique_equations += n_unique
    compile_s = time.perf_counter() - t0

    return SpinUniqueCommutator(
        rank=rank,
        functions=tuple(compiled),
        blocks=tuple(unique_key for unique_key, _ in compiled),
        generation_s={
            "build": build_s,
            "commutator_build": commutator_build_s,
            "contract": contract_s,
            "to_manybody_equation": equation_s,
            "compile": compile_s,
            "total": build_s + commutator_build_s + contract_s + equation_s + compile_s,
            "mapped_equations": n_mapped_equations,
        },
        n_equations=n_equations,
        n_expression_terms=len(expression),
        n_unique_equations=n_unique_equations,
    )


def excitation_component(rank, nalpha):
    return " ".join(
        ["v+"] * nalpha
        + ["V+"] * (rank - nalpha)
        + ["O"] * (rank - nalpha)
        + ["o"] * nalpha
    )


def is_alpha_space(space):
    return space in "ov"


def is_spin_conserving(cre_spaces, ann_spaces):
    return sum(is_alpha_space(space) for space in cre_spaces) == sum(
        is_alpha_space(space) for space in ann_spaces
    )


def spin_conserving_x_operator(rank):
    x_op = w.op("E_0", [""])
    seen_terms = set()
    for body_rank in range(1, rank + 1):
        for cre_spaces in itertools.product(SPACES, repeat=body_rank):
            for ann_spaces in itertools.product(SPACES, repeat=body_rank):
                if not is_spin_conserving(cre_spaces, ann_spaces):
                    continue
                component = " ".join(
                    [f"{space}+" for space in cre_spaces] + list(ann_spaces)
                )
                term = w.op("x", [component], unique=False)
                signature = str(term)
                if signature in seen_terms:
                    continue
                seen_terms.add(signature)
                x_op = x_op + term
    return x_op


def make_spin_integrated_commutator(rank, nocc, nvir):
    t0 = time.perf_counter()
    w.reset_space()
    for space in SPACES:
        w.add_space(space, "fermion", SPACE_KIND[space], SPACE_INDICES[space])

    x_op = spin_conserving_x_operator(rank)

    t_op = None
    for body_rank in range(1, rank + 1):
        for nalpha in range(body_rank + 1):
            term = w.op("t", [excitation_component(body_rank, nalpha)], unique=False)
            t_op = term if t_op is None else t_op + term
    a_op = t_op - t_op.adjoint()
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    commutator_expr = w.commutator(x_op, a_op)
    commutator_build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    wt = w.WickTheorem()
    wt.set_single_threaded(True)
    expression = wt.contract(commutator_expr, 0, 2 * rank).canonicalize()
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

    return SpinIntegratedCommutator(
        rank=rank,
        functions=tuple(compiled),
        blocks=tuple(result_key for result_key, _ in compiled),
        generation_s={
            "build": build_s,
            "commutator_build": commutator_build_s,
            "contract": contract_s,
            "to_manybody_equation": equation_s,
            "compile": compile_s,
            "total": build_s + commutator_build_s + contract_s + equation_s + compile_s,
        },
        n_equations=sum(
            len(compiled_einsum_statements(equation))
            for equations in many_body_equations.values()
            for equation in equations
        ),
        n_expression_terms=len(expression),
    )


def zero_tensors(keys, nocc, nvir):
    tensors = {}
    for key in keys:
        tensors[key] = (
            np.array(0.0) if key == "" else np.zeros(tensor_shape(key, nocc, nvir))
        )
    return tensors


def spin_integrated_hamiltonian_tensors(data, rank, keys):
    so_h0, _, occ, virt, _ = normal_ordered_hamiltonian_tensors(data, rank)
    nocc = len(occ) // 2
    nvir = len(virt) // 2
    spin_slices = {
        "o": np.arange(0, 2 * nocc, 2),
        "O": np.arange(1, 2 * nocc, 2),
        "v": np.arange(0, 2 * nvir, 2),
        "V": np.arange(1, 2 * nvir, 2),
    }

    tensors = zero_tensors(keys, nocc, nvir)
    for key in list(tensors):
        if key == "":
            tensors[key] = so_h0[""].copy()
            continue
        body_rank = len(key) // 2
        if body_rank > 2:
            continue
        base_key = "".join("o" if space in "oO" else "v" for space in key)
        block = so_h0[base_key]
        slices = [spin_slices[space] for space in key]
        tensors[key][...] = block[np.ix_(*slices)]
    return tensors, nocc, nvir


def unique_tensor_shape(unique_key, nocc, nvir):
    if unique_key == "":
        return ()
    _, base_key = unique_key.split(":", 1)
    return tensor_shape(base_key, nocc, nvir)


def zero_unique_tensors(keys, nocc, nvir):
    tensors = {}
    for key in keys:
        tensors[key] = (
            np.array(0.0)
            if key == ""
            else np.zeros(unique_tensor_shape(key, nocc, nvir))
        )
    return tensors


def all_spin_unique_x_keys(rank):
    keys = [""]
    for body_rank in range(1, rank + 1):
        for nalpha in unique_spin_counts(body_rank):
            label = spin_count_label(body_rank, nalpha)
            for base_key in (
                "".join(item) for item in itertools.product("ov", repeat=2 * body_rank)
            ):
                keys.append(f"{label}:{base_key}")
    return tuple(keys)


def spin_unique_hamiltonian_tensors(data, rank, keys):
    so_h0, _, occ, virt, _ = normal_ordered_hamiltonian_tensors(data, rank)
    nocc = len(occ) // 2
    nvir = len(virt) // 2
    spin_slices = {
        "o": np.arange(0, 2 * nocc, 2),
        "O": np.arange(1, 2 * nocc, 2),
        "v": np.arange(0, 2 * nvir, 2),
        "V": np.arange(1, 2 * nvir, 2),
    }

    tensors = zero_unique_tensors(keys, nocc, nvir)
    for unique_key in list(tensors):
        if unique_key == "":
            tensors[unique_key] = so_h0[""].copy()
            continue

        label, base_key = unique_key.split(":", 1)
        body_rank = len(base_key) // 2
        if body_rank > 2:
            continue

        nalpha = spin_count_from_label(label, body_rank)
        ann = base_key[:body_rank]
        cre = base_key[body_rank:]
        full_key = (
            ann[:nalpha] + ann[nalpha:].upper() + cre[:nalpha] + cre[nalpha:].upper()
        )
        spinorbital_key = full_key.lower()
        slices = [spin_slices[space] for space in full_key]
        tensors[unique_key][...] = so_h0[spinorbital_key][np.ix_(*slices)]
    return tensors, nocc, nvir


def grouped_permutations(groups):
    options = []
    for positions, values in groups:
        group_options = []
        size = len(values)
        for perm in itertools.permutations(range(size)):
            permuted_values = tuple(values[idx] for idx in perm)
            group_options.append((positions, permuted_values, permutation_parity(perm)))
        options.append(group_options)
    return itertools.product(*options)


def add_spin_integrated_amplitude(tensor, groups, value):
    for choices in grouped_permutations(groups):
        indices = [0] * tensor.ndim
        sign = 1
        for positions, values, group_sign in choices:
            sign *= group_sign
            for position, index in zip(positions, values):
                indices[position] = index
        tensor[tuple(indices)] += sign * value


def excitation_tensor_key_and_indices(excitation, nocc_spatial):
    rank = excitation["rank"]
    ann_alpha = tuple(i for i, spin in excitation["ann"] if spin == "a")
    ann_beta = tuple(i for i, spin in excitation["ann"] if spin == "b")
    cre_alpha = tuple(a - nocc_spatial for a, spin in excitation["cre"] if spin == "a")
    cre_beta = tuple(a - nocc_spatial for a, spin in excitation["cre"] if spin == "b")
    nalpha = len(ann_alpha)

    ann_key = "o" * nalpha + "O" * (rank - nalpha)
    cre_key = "v" * nalpha + "V" * (rank - nalpha)
    key = ann_key + cre_key
    indices = ann_alpha + ann_beta + cre_alpha + cre_beta

    groups = [
        (tuple(range(0, nalpha)), ann_alpha),
        (tuple(range(nalpha, rank)), ann_beta),
        (tuple(range(rank, rank + nalpha)), cre_alpha),
        (tuple(range(rank + nalpha, 2 * rank)), cre_beta),
    ]
    return key, indices, groups


def amplitudes_to_tensors(excitations, amplitudes, nocc, nvir, rank):
    keys = []
    for body_rank in range(1, rank + 1):
        for nalpha in range(body_rank + 1):
            ann_key = "o" * nalpha + "O" * (body_rank - nalpha)
            cre_key = "v" * nalpha + "V" * (body_rank - nalpha)
            keys.append(ann_key + cre_key)
            keys.append(cre_key + ann_key)
    tensors = zero_tensors(keys, nocc, nvir)

    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) <= SCREEN:
            continue
        key, _, groups = excitation_tensor_key_and_indices(excitation, nocc)
        add_spin_integrated_amplitude(tensors[key], groups, amplitude)

    for body_rank in range(1, rank + 1):
        for nalpha in range(body_rank + 1):
            ann_key = "o" * nalpha + "O" * (body_rank - nalpha)
            cre_key = "v" * nalpha + "V" * (body_rank - nalpha)
            exc_key = ann_key + cre_key
            deexc_key = cre_key + ann_key
            axes = tuple(range(body_rank, 2 * body_rank)) + tuple(range(body_rank))
            tensors[deexc_key][...] = np.transpose(tensors[exc_key], axes)
    return tensors


def unique_excitation_key_indices_groups(excitation, nocc_spatial):
    full_key, indices, _ = excitation_tensor_key_and_indices(excitation, nocc_spatial)
    dummy_indices = "".join(
        chr(ord("a") + pos) for pos in range(2 * excitation["rank"])
    )
    sign, unique_name, unique_base, unique_dummy = spin_unique_tensor_ref(
        "t", full_key, dummy_indices
    )
    order = tuple(dummy_indices.index(label) for label in unique_dummy)
    unique_indices = tuple(indices[pos] for pos in order)
    label = unique_name.split(":", 1)[1]
    body_rank = excitation["rank"]
    nalpha = spin_count_from_label(label, body_rank)
    groups = [
        (tuple(range(0, nalpha)), unique_indices[:nalpha]),
        (tuple(range(nalpha, body_rank)), unique_indices[nalpha:body_rank]),
        (
            tuple(range(body_rank, body_rank + nalpha)),
            unique_indices[body_rank : body_rank + nalpha],
        ),
        (
            tuple(range(body_rank + nalpha, 2 * body_rank)),
            unique_indices[body_rank + nalpha :],
        ),
    ]
    return f"{label}:{unique_base}", unique_indices, groups, sign


def spin_unique_excitation_maps(excitations, nocc_spatial):
    return tuple(
        unique_excitation_key_indices_groups(excitation, nocc_spatial)
        for excitation in excitations
    )


def add_spin_unique_amplitude(sum_tensor, count_tensor, groups, value):
    for choices in grouped_permutations(groups):
        indices = [0] * sum_tensor.ndim
        sign = 1
        for positions, values, group_sign in choices:
            sign *= group_sign
            for position, index in zip(positions, values):
                indices[position] = index
        index_tuple = tuple(indices)
        sum_tensor[index_tuple] += sign * value
        count_tensor[index_tuple] += 1.0


def spin_unique_t_keys(rank):
    keys = []
    for body_rank in range(1, rank + 1):
        for nalpha in unique_spin_counts(body_rank):
            label = spin_count_label(body_rank, nalpha)
            exc_base = "o" * body_rank + "v" * body_rank
            deexc_base = "v" * body_rank + "o" * body_rank
            keys.append(f"{label}:{exc_base}")
            keys.append(f"{label}:{deexc_base}")
    return tuple(keys)


def amplitudes_to_spin_unique_tensors(excitations, amplitudes, nocc, nvir, rank):
    excitation_maps = spin_unique_excitation_maps(excitations, nocc)
    return amplitudes_to_spin_unique_tensors_from_maps(
        excitation_maps, amplitudes, nocc, nvir, rank
    )


def amplitudes_to_spin_unique_tensors_from_maps(
    excitation_maps, amplitudes, nocc, nvir, rank
):
    keys = spin_unique_t_keys(rank)
    tensors = zero_unique_tensors(keys, nocc, nvir)
    counts = zero_unique_tensors(keys, nocc, nvir)

    for (unique_key, _, groups, map_sign), amplitude in zip(
        excitation_maps, amplitudes
    ):
        if abs(amplitude) <= SCREEN:
            continue
        add_spin_unique_amplitude(
            tensors[unique_key], counts[unique_key], groups, map_sign * amplitude
        )

    for key in keys:
        nonzero = counts[key] > 0
        if np.any(nonzero):
            tensors[key][nonzero] /= counts[key][nonzero]

    for body_rank in range(1, rank + 1):
        axes = tuple(range(body_rank, 2 * body_rank)) + tuple(range(body_rank))
        for nalpha in unique_spin_counts(body_rank):
            label = spin_count_label(body_rank, nalpha)
            exc_key = f"{label}:{'o' * body_rank}{'v' * body_rank}"
            deexc_key = f"{label}:{'v' * body_rank}{'o' * body_rank}"
            tensors[deexc_key][...] = np.transpose(tensors[exc_key], axes)
    return tensors


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
        for space in SPACES:
            positions = [
                offset + idx for idx, label in enumerate(spaces) if label == space
            ]
            result = antisymmetrize_positions(result, positions)
    return result


def antisymmetrize_tensors(tensors):
    return {key: antisymmetrize_block(key, value) for key, value in tensors.items()}


def antisymmetrize_unique_block(unique_key, tensor):
    if unique_key == "" or ":" not in unique_key:
        return tensor

    label, base_key = unique_key.split(":", 1)
    rank = len(base_key) // 2
    if rank < 2:
        return tensor

    nalpha = spin_count_from_label(label, rank)
    result = tensor
    for start, stop in (
        (0, nalpha),
        (nalpha, rank),
        (rank, rank + nalpha),
        (rank + nalpha, 2 * rank),
    ):
        positions_by_space = {}
        for position in range(start, stop):
            positions_by_space.setdefault(base_key[position], []).append(position)
        for positions in positions_by_space.values():
            result = antisymmetrize_positions(result, tuple(positions))
    return result


def antisymmetrize_unique_tensors(tensors):
    return {
        key: antisymmetrize_unique_block(key, value) for key, value in tensors.items()
    }


def evaluate_commutator(commutator, x, t):
    y = {}
    for key, function in commutator.functions:
        y[key] = function(x, t)
    return y


def complete_x_blocks(tensors, keys, nocc, nvir):
    completed = zero_tensors(keys, nocc, nvir)
    for key, value in tensors.items():
        if key in completed:
            completed[key][...] = value
    return completed


def complete_unique_x_blocks(tensors, keys, nocc, nvir):
    completed = zero_unique_tensors(keys, nocc, nvir)
    for key, value in tensors.items():
        if key in completed:
            completed[key][...] = value
    return completed


def add_scaled_tensors(target, source, scale):
    for key, value in source.items():
        target[key] += scale * value


def tensor_norm(tensors):
    return math.sqrt(
        sum(float(np.vdot(value, value).real) for value in tensors.values())
    )


def bch_hbar(h0, t, commutator, nocc, nvir, max_comm=DEFAULT_MAX_COMM):
    hbar = {key: value.copy() for key, value in h0.items()}
    nested = {key: value.copy() for key, value in h0.items()}
    commutator_norms = []

    for ncomm in range(1, max_comm + 1):
        nested = complete_x_blocks(
            antisymmetrize_tensors(evaluate_commutator(commutator, nested, t)),
            commutator.blocks,
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


def bch_hbar_spin_unique(h0, t, commutator, nocc, nvir, max_comm=DEFAULT_MAX_COMM):
    x_keys = all_spin_unique_x_keys(commutator.rank)
    hbar = {key: value.copy() for key, value in h0.items()}
    nested = {key: value.copy() for key, value in h0.items()}
    commutator_norms = []

    for ncomm in range(1, max_comm + 1):
        nested = complete_unique_x_blocks(
            antisymmetrize_unique_tensors(evaluate_commutator(commutator, nested, t)),
            x_keys,
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


def excitation_values(tensors, excitations, nocc):
    values = np.zeros(len(excitations))
    for idx, excitation in enumerate(excitations):
        key, indices, _ = excitation_tensor_key_and_indices(excitation, nocc)
        values[idx] = tensors[key][indices]
    return values


def spin_unique_excitation_values(tensors, excitations, nocc):
    excitation_maps = spin_unique_excitation_maps(excitations, nocc)
    return spin_unique_excitation_values_from_maps(tensors, excitation_maps)


def spin_unique_excitation_values_from_maps(tensors, excitation_maps):
    values = np.zeros(len(excitation_maps))
    for idx, (unique_key, unique_indices, _, map_sign) in enumerate(excitation_maps):
        values[idx] = map_sign * tensors[unique_key][unique_indices]
    return values


def solve_spin_integrated_dsrg(
    data,
    rank,
    flow_param=DEFAULT_FLOW_PARAM,
    e_tol=DEFAULT_E_TOL,
    r_tol=DEFAULT_R_TOL,
    max_iter=DEFAULT_MAX_ITER,
    iteration_log_path=None,
    print_iterations=False,
):
    nocc = data.rhf.na
    nvir = data.rhf.nmo - data.rhf.na
    commutator = make_spin_unique_commutator(rank, nocc, nvir)
    h0, nocc, nvir = spin_unique_hamiltonian_tensors(
        data, rank, all_spin_unique_x_keys(rank)
    )
    excitations = enumerate_spin_conserving_excitations(
        data.rhf.nmo, data.rhf.na, data.eps, rank
    )
    for excitation in excitations:
        excitation["regularized_denominator"] = regularized_denominator(
            excitation["denom"], flow_param
        )
    excitation_maps = spin_unique_excitation_maps(excitations, nocc)

    h0_offdiag = spin_unique_excitation_values_from_maps(h0, excitation_maps)
    amplitudes = np.array(
        [
            h0_offdiag[idx] * excitation["regularized_denominator"]
            for idx, excitation in enumerate(excitations)
        ]
    )
    diis = DIIS(diis_start=3, diis_nvec=8, diis_min=3, do_diis=True)
    history = []
    previous_energy = None
    t0 = time.perf_counter()

    for iteration in range(max_iter + 1):
        iter_t0 = time.perf_counter()
        t_tensors = amplitudes_to_spin_unique_tensors_from_maps(
            excitation_maps, amplitudes, nocc, nvir, rank
        )
        hbar, commutator_norms = bch_hbar_spin_unique(
            h0, t_tensors, commutator, nocc, nvir
        )
        energy = float(hbar[""])
        hbar_offdiag = spin_unique_excitation_values_from_maps(hbar, excitation_maps)
        fixed_point = np.array(
            [
                (hbar_offdiag[idx] + excitation["denom"] * amplitudes[idx])
                * excitation["regularized_denominator"]
                for idx, excitation in enumerate(excitations)
            ]
        )
        update = fixed_point - amplitudes
        rms_update = float(np.linalg.norm(update))
        delta_energy = 0.0 if previous_energy is None else energy - previous_energy
        history.append(
            {
                "natoms": data.natoms,
                "spacing": data.spacing,
                "rank": rank,
                "iteration": iteration,
                "energy": energy,
                "delta_energy": delta_energy,
                "rms_update": rms_update,
                "ncomm": len(commutator_norms),
                "iter_s": time.perf_counter() - iter_t0,
                "diis_status": getattr(diis, "status", ""),
            }
        )
        if iteration_log_path is not None:
            log_path = Path(iteration_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(history[-1], sort_keys=True) + "\n")
        if print_iterations:
            print(
                "ITER "
                f"H{data.natoms} DSRG({rank}) "
                f"it={iteration:3d} "
                f"E={energy:.15f} "
                f"dE={delta_energy:.3e} "
                f"rms={rms_update:.3e} "
                f"ncomm={len(commutator_norms):2d} "
                f"iter_s={history[-1]['iter_s']:.2f} "
                f"diis={getattr(diis, 'status', '')}",
                flush=True,
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
                    "n_unique_equations": commutator.n_unique_equations,
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
            "n_unique_equations": commutator.n_unique_equations,
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
        lookup[(row.get("natoms"), row.get("spacing"), row.get("rank"))] = row
    return lookup


def run_case(
    natoms,
    spacing,
    rank,
    sparse_lookup,
    iteration_log_path=None,
    print_iterations=False,
):
    logger.set_verbosity_level(0)
    with contextlib.redirect_stdout(io.StringIO()):
        data = build_linear_h_data(natoms, spacing)
        fci = forte2_fci_energy(data)

    result = solve_spin_integrated_dsrg(
        data,
        rank,
        iteration_log_path=iteration_log_path,
        print_iterations=print_iterations,
    )
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
        "fci": fci,
        "spin_integrated_energy": result["energy"],
        "reference_sparse_energy": reference_energy,
        "spin_integrated_minus_reference_sparse": (
            None if reference_energy is None else result["energy"] - reference_energy
        ),
        "spin_integrated_minus_fci": result["energy"] - fci["energy"],
        "spin_integrated_solve_s": result["solve_s"],
        "spin_integrated_iterations": result["iterations"],
        "n_amplitudes": result["n_amplitudes"],
        "spin_integrated_history_tail": result["history_tail"],
        "spin_integrated_equations": result["equations"],
        "sparse_reference": {
            "energy": sparse_row.get("dsrg_energy", reference_energy),
            "solve_s": sparse_row.get("solve_s"),
            "iterations": sparse_row.get("iterations"),
        },
        "completed_at": now_s(),
    }


def run_case_with_timeout(
    natoms,
    spacing,
    rank,
    sparse_lookup,
    timeout_s,
    iteration_log_path=None,
    print_iterations=False,
):
    if timeout_s is None or timeout_s <= 0.0:
        return run_case(
            natoms,
            spacing,
            rank,
            sparse_lookup,
            iteration_log_path,
            print_iterations,
        )

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(
        target=lambda q: q.put(
            run_case(
                natoms,
                spacing,
                rank,
                sparse_lookup,
                iteration_log_path,
                print_iterations,
            )
        ),
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
        "--output", default="experiments/wickd_dsrg_spin_integrated_results.json"
    )
    parser.add_argument(
        "--sparse-results", default="experiments/dsrg_hchain_results.json"
    )
    parser.add_argument("--atoms", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--spacing", type=float, default=0.74)
    parser.add_argument("--case-timeout", type=float, default=300.0)
    parser.add_argument("--iteration-log", default=None)
    parser.add_argument("--print-iterations", action="store_true")
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
            "method": "Wick&d-generated spin-unique alpha/beta DSRG(n)",
            "iteration_log": args.iteration_log,
        },
        "cases": [],
    }
    save_results(Path(args.output), payload)

    for natoms in args.atoms:
        for rank in args.ranks:
            print(
                f"START {now_s()} H{natoms} DSRG({rank}) spin-integrated Wick&d",
                flush=True,
            )
            row = run_case_with_timeout(
                natoms,
                args.spacing,
                rank,
                sparse_lookup,
                args.case_timeout,
                args.iteration_log,
                args.print_iterations,
            )
            payload["cases"].append(row)
            save_results(Path(args.output), payload)
            diff = row.get("spin_integrated_minus_reference_sparse")
            diff_text = "n/a" if diff is None else f"{diff:.3e}"
            energy = row.get("spin_integrated_energy")
            energy_text = "n/a" if energy is None else f"{energy:.15f}"
            print(
                f"DONE H{natoms} DSRG({rank}) status={row['status']} "
                f"E={energy_text} diff_sparse={diff_text} "
                f"iters={row.get('spin_integrated_iterations', 'n/a')} "
                f"solve={row.get('spin_integrated_solve_s', row.get('elapsed_s', 0.0)):.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
