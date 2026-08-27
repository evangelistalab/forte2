"""Benchmark the dedicated cumulant Wick engine against the sparse GNO engine."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import forte2
from experiments.fused_gno_commutator_benchmark import coefficient_error, make_case
from forte2.dsrg.sparse_mrdsrg2 import _make_antihermitian_cluster_operator
from forte2.lib import sparse_ops


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def benchmark_commutator(case, repeats, screen_thresh):
    max_cumulant = 3
    lhs = sparse_ops.generalized_normal_order(
        case["hamiltonian"],
        case["vacuum"],
        case["norb"],
        max_cumulant=max_cumulant,
        max_rank=2,
        screen_thresh=screen_thresh,
    )
    amplitudes = np.array(
        [
            0.01 * np.sin(index + 1.0) / (1.0 + excitation.rank)
            for index, excitation in enumerate(case["excitations"])
        ]
    )
    rhs = _make_antihermitian_cluster_operator(
        case["vacuum"],
        case["norb"],
        max_cumulant,
        case["excitations"],
        amplitudes,
        screen_thresh,
    )
    reference, construction_s = timed(
        lambda: sparse_ops.CumulantReference(
            case["vacuum"],
            case["norb"],
            max_cumulant=max_cumulant,
            screen_thresh=screen_thresh,
        )
    )
    engine = sparse_ops.CumulantWickEngine(reference, 2, screen_thresh)

    sparse_times = []
    cumulant_times = []
    sparse_result = None
    cumulant_result = None
    for _ in range(repeats):
        sparse_result, elapsed = timed(
            lambda: lhs.commutator(rhs, max_rank=2, screen_thresh=screen_thresh)
        )
        sparse_times.append(elapsed)
        cumulant_result, elapsed = timed(lambda: engine.commutator(lhs, rhs))
        cumulant_times.append(elapsed)

    return {
        "nmo": case["norb"],
        "n_excitations": len(case["excitations"]),
        "terms": {"lhs": len(lhs), "rhs": len(rhs), "result": len(sparse_result)},
        "seconds": {
            "reference_construction": construction_s,
            "sparse_min": min(sparse_times),
            "sparse_median": float(np.median(sparse_times)),
            "cumulant_min": min(cumulant_times),
            "cumulant_median": float(np.median(cumulant_times)),
        },
        "speedup_excluding_construction": min(sparse_times) / min(cumulant_times),
        "coefficient_error": coefficient_error(cumulant_result, sparse_result),
    }


def benchmark_solver(case, maxiter, repeats, screen_thresh):
    results = {}
    for backend in ("sparse", "cumulant"):
        times = []
        result = None
        for _ in range(repeats):
            result, elapsed = timed(
                lambda: forte2.solve_sparse_mrdsrg2(
                    case["hamiltonian"],
                    case["vacuum"],
                    case["norb"],
                    case["excitations"],
                    flow_param=5.0,
                    max_cumulant=3,
                    model_space=case["model_space"],
                    screen_thresh=screen_thresh,
                    maxiter=maxiter,
                    max_commutators=4,
                    do_diis=False,
                    gno_backend=backend,
                )
            )
            times.append(elapsed)
        results[backend] = {
            "seconds_min": min(times),
            "seconds_median": float(np.median(times)),
            "energy": result.energy,
            "iterations": result.iterations,
            "converged": result.converged,
        }
    results["speedup"] = (
        results["sparse"]["seconds_min"] / results["cumulant"]["seconds_min"]
    )
    results["energy_error"] = abs(
        results["sparse"]["energy"] - results["cumulant"]["energy"]
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--screen-thresh", type=float, default=1.0e-12)
    parser.add_argument("--solver-iterations", type=int, default=0)
    args = parser.parse_args()

    case = make_case(2, args.screen_thresh)
    result = {
        "commutator": benchmark_commutator(case, args.repeats, args.screen_thresh)
    }
    if args.solver_iterations > 0:
        result["solver"] = benchmark_solver(
            case,
            args.solver_iterations,
            args.repeats,
            args.screen_thresh,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
