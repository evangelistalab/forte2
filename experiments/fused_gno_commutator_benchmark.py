"""Benchmark fused generalized-normal-ordered commutators on H2/cc-pVDZ."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import forte2
from experiments import dsrg_hchain_benchmark as hchain
from experiments.h2_ccpvdz_mr_sr_reap_sow import cas22_reference
from forte2.dsrg import sparse_mrdsrg2 as mrdsrg
from forte2.dsrg.sparse_mrdsrg2 import _make_antihermitian_cluster_operator
from forte2.helpers import logger
from forte2.lib import sparse_ops


def coefficient_error(lhs, rhs) -> float:
    lhs_terms = {term.str(): coefficient for term, coefficient in lhs}
    rhs_terms = {term.str(): coefficient for term, coefficient in rhs}
    keys = set(lhs_terms) | set(rhs_terms)
    return float(
        np.sqrt(
            sum(
                abs(lhs_terms.get(key, 0.0) - rhs_terms.get(key, 0.0)) ** 2
                for key in keys
            )
        )
    )


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def make_case(max_rank: int, screen_thresh: float) -> dict:
    hchain.BASIS = "cc-pVDZ"
    logger.set_verbosity_level(0)
    system, rhf, hamiltonian, orbital_energies = (
        hchain.build_linear_h_sparse_hamiltonian(2, 0.75)
    )
    del system
    vacuum, model_space, _ = cas22_reference(hamiltonian, rhf.nmo)
    max_cumulant = min(max_rank, 4)
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=[0, 1],
        virtual_orbitals=list(range(2, rhf.nmo)),
        orbital_energies=orbital_energies,
        max_rank=max_rank,
    )
    amplitudes = np.array(
        [
            0.01 * np.sin(index + 1.0) / (1.0 + excitation.rank)
            for index, excitation in enumerate(excitations)
        ]
    )
    lhs = sparse_ops.generalized_normal_order(
        hamiltonian,
        vacuum,
        rhf.nmo,
        max_cumulant=max_cumulant,
        screen_thresh=screen_thresh,
        max_rank=max_rank,
    )
    rhs = _make_antihermitian_cluster_operator(
        vacuum,
        rhf.nmo,
        max_cumulant,
        excitations,
        amplitudes,
        screen_thresh,
    )
    return {
        "hamiltonian": hamiltonian,
        "norb": rhf.nmo,
        "vacuum": vacuum,
        "model_space": model_space,
        "max_cumulant": max_cumulant,
        "excitations": excitations,
        "lhs": lhs,
        "rhs": rhs,
    }


def benchmark(case: dict, max_rank: int, repeats: int, screen_thresh: float) -> dict:
    lhs = case["lhs"]
    rhs = case["rhs"]
    vacuum = case["vacuum"]
    norb = case["norb"]
    max_cumulant = case["max_cumulant"]
    excitations = case["excitations"]

    lhs_bare, lhs_expand_s = timed(lambda: lhs.to_sparse_operator(screen_thresh))
    rhs_bare, rhs_expand_s = timed(lambda: rhs.to_sparse_operator(screen_thresh))
    bare_commutator, bare_commutator_s = timed(lambda: lhs_bare.commutator(rhs_bare))
    reference, gno_s = timed(
        lambda: sparse_ops.generalized_normal_order(
            bare_commutator,
            vacuum,
            norb,
            max_cumulant=max_cumulant,
            screen_thresh=screen_thresh,
            max_rank=max_rank,
        )
    )

    fused_times = []
    fused = None
    for _ in range(repeats):
        fused, elapsed = timed(
            lambda: lhs.commutator(rhs, max_rank=max_rank, screen_thresh=screen_thresh)
        )
        fused_times.append(elapsed)
    assert fused is not None

    reference_s = lhs_expand_s + rhs_expand_s + bare_commutator_s + gno_s
    return {
        "max_rank": max_rank,
        "nmo": norb,
        "n_excitations": len(excitations),
        "terms": {
            "lhs_gno": len(lhs),
            "rhs_gno": len(rhs),
            "lhs_bare": len(lhs_bare),
            "rhs_bare": len(rhs_bare),
            "bare_commutator": len(bare_commutator),
            "result": len(reference),
        },
        "seconds": {
            "lhs_expand": lhs_expand_s,
            "rhs_expand": rhs_expand_s,
            "bare_commutator": bare_commutator_s,
            "gno": gno_s,
            "reference_total": reference_s,
            "fused_min": min(fused_times),
            "fused_median": float(np.median(fused_times)),
        },
        "speedup": reference_s / min(fused_times),
        "coefficient_error": coefficient_error(fused, reference),
    }


def old_gno_commutator(
    lhs,
    rhs,
    vacuum,
    norb,
    max_cumulant,
    max_rank,
    screen_thresh,
):
    bare = lhs.to_sparse_operator(screen_thresh).commutator(
        rhs.to_sparse_operator(screen_thresh)
    )
    return sparse_ops.generalized_normal_order(
        bare,
        vacuum,
        norb,
        max_cumulant=max_cumulant,
        screen_thresh=screen_thresh,
        max_rank=max_rank,
    )


def benchmark_solver(
    case: dict, max_rank: int, maxiter: int, screen_thresh: float
) -> dict:
    solver = getattr(forte2, f"solve_sparse_mrdsrg{max_rank}")
    original_commutator = mrdsrg._gno_commutator
    results = {}
    try:
        for name, commutator in (
            ("reference", old_gno_commutator),
            ("fused", original_commutator),
        ):
            mrdsrg._gno_commutator = commutator
            result, seconds = timed(
                lambda: solver(
                    case["hamiltonian"],
                    case["vacuum"],
                    case["norb"],
                    case["excitations"],
                    flow_param=5.0,
                    max_cumulant=case["max_cumulant"],
                    model_space=case["model_space"],
                    screen_thresh=screen_thresh,
                    maxiter=maxiter,
                    max_commutators=4,
                    do_diis=False,
                )
            )
            results[name] = {
                "seconds": seconds,
                "energy": result.energy,
                "iterations": result.iterations,
                "converged": result.converged,
            }
    finally:
        mrdsrg._gno_commutator = original_commutator
    results["speedup"] = results["reference"]["seconds"] / results["fused"]["seconds"]
    results["energy_error"] = abs(
        results["reference"]["energy"] - results["fused"]["energy"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--screen-thresh", type=float, default=1.0e-12)
    parser.add_argument("--solver-iterations", type=int, default=0)
    args = parser.parse_args()
    case = make_case(args.rank, args.screen_thresh)
    result = {
        "commutator": benchmark(case, args.rank, args.repeats, args.screen_thresh)
    }
    if args.solver_iterations > 0:
        result["solver"] = benchmark_solver(
            case, args.rank, args.solver_iterations, args.screen_thresh
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
