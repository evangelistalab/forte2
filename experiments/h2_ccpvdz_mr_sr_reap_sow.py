"""Restartable reap/sow driver for H2 SR- and MR-LDSRG(n) sweeps."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

import forte2
from experiments import dsrg_hchain_benchmark as sr_benchmark
from forte2.helpers import logger
from forte2.lib.det import Determinant
from forte2.lib.sparse_ops import (
    CumulantReference,
    SparseState,
    overlap,
    sparse_operator_hamiltonian,
)

BASIS = "cc-pVDZ"
BOND_LENGTHS = (0.75, 1.50, 2.25, 3.00)
RANKS = (2, 3, 4)
EXPONENTS = tuple(-4.0 + 0.5 * index for index in range(15))
E_TOL = 1.0e-10
R_TOL = 1.0e-5
MAX_ITER = 80


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def number_token(value: float) -> str:
    return f"{value:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")


def case_id(case: dict) -> str:
    bond = f"{case['bond_length']:.2f}".replace(".", "p")
    return (
        f"{case['method']}_R{bond}_r{case['rank']}_N"
        f"{number_token(case['flow_exponent'])}"
    )


def make_manifest() -> dict:
    cases = []
    for rank in RANKS:
        for bond_length in BOND_LENGTHS:
            for exponent in EXPONENTS:
                for method in ("sr", "mr"):
                    case = {
                        "method": method,
                        "bond_length": bond_length,
                        "rank": rank,
                        "flow_exponent": exponent,
                        "flow_param": 10.0**exponent,
                    }
                    case["id"] = case_id(case)
                    cases.append(case)
    return {
        "metadata": {
            "created_at": now_s(),
            "basis": BASIS,
            "bond_lengths_angstrom": list(BOND_LENGTHS),
            "ranks": list(RANKS),
            "mr_max_cumulants": {str(rank): max(3, min(rank, 4)) for rank in RANKS},
            "flow_exponents": list(EXPONENTS),
            "e_tol": E_TOL,
            "r_tol": R_TOL,
            "max_iter": MAX_ITER,
            "mr_reference": "CAS(2e,2o) in generalized-Fock semicanonical orbitals",
        },
        "cases": cases,
    }


def sow(root: Path, reset_locks: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(exist_ok=True)
    (root / "locks").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        atomic_json(manifest_path, make_manifest())
    if reset_locks:
        for lock in (root / "locks").iterdir():
            if lock.is_dir():
                lock.rmdir()
    print(f"SOWED {len(json.loads(manifest_path.read_text())['cases'])} cases")


def determinant_label(occupations: str, norb: int) -> str:
    return occupations + "0" * (norb - len(occupations))


def cas22_reference(hamiltonian, norb: int):
    model_space = tuple(
        Determinant(determinant_label(label, norb))
        for label in ("20", "ab", "ba", "02")
    )
    matrix = np.array(
        [
            [
                overlap(
                    SparseState({bra: 1.0}),
                    hamiltonian.apply_to_state(SparseState({ket: 1.0})),
                )
                for ket in model_space
            ]
            for bra in model_space
        ],
        dtype=complex,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    vacuum = SparseState(
        {
            determinant: coefficient
            for determinant, coefficient in zip(model_space, eigenvectors[:, 0])
            if abs(coefficient) > 1.0e-14
        }
    )
    return vacuum, model_space, float(eigenvalues[0].real)


def semicanonical_h2_problem(system, rhf, hamiltonian):
    vacuum, _, _ = cas22_reference(hamiltonian, rhf.nmo)
    reference = CumulantReference(vacuum, rhf.nmo, max_cumulant=3)
    gamma = np.array(
        [
            [
                sum(reference.gamma(p, alpha, q, alpha).real for alpha in (True, False))
                for q in range(rhf.nmo)
            ]
            for p in range(rhf.nmo)
        ]
    )
    coefficients = rhf.C[0]
    hcore = np.einsum(
        "pq,pi,qj->ij", system.ints_hcore(), coefficients, coefficients, optimize=True
    )
    eri = system.fock_builder.two_electron_integrals_block(coefficients)
    fock = hcore + np.einsum("rs,prqs->pq", gamma, eri, optimize=True)
    fock -= 0.5 * np.einsum("rs,prsq->pq", gamma, eri, optimize=True)

    rotation = np.zeros_like(fock)
    for orbital_slice in (slice(0, 2), slice(2, rhf.nmo)):
        _, vectors = np.linalg.eigh(fock[orbital_slice, orbital_slice])
        rotation[orbital_slice, orbital_slice] = vectors
    orbital_energies = np.diag(rotation.T @ fock @ rotation)
    hcore = rotation.T @ hcore @ rotation
    eri = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        rotation,
        rotation,
        eri,
        rotation,
        rotation,
        optimize=True,
    )
    hamiltonian = sparse_operator_hamiltonian(system.nuclear_repulsion, hcore, eri)
    vacuum, model_space, reference_energy = cas22_reference(hamiltonian, rhf.nmo)
    return hamiltonian, vacuum, model_space, reference_energy, orbital_energies


def run_sr(case: dict) -> dict:
    sr_benchmark.BASIS = BASIS
    args = (
        2,
        case["bond_length"],
        case["rank"],
        case["flow_param"],
        E_TOL,
        R_TOL,
        MAX_ITER,
        True,
        3,
        8,
        3,
    )
    return sr_benchmark.run_case(args)


def run_mr(case: dict) -> dict:
    sr_benchmark.BASIS = BASIS
    logger.set_verbosity_level(0)
    with contextlib.redirect_stdout(io.StringIO()):
        system, rhf, hamiltonian, eps = sr_benchmark.build_linear_h_sparse_hamiltonian(
            2, case["bond_length"]
        )
        fci = sr_benchmark.forte2_fci_energy(system, rhf, 2)

    hamiltonian, vacuum, _, reference_energy, eps = semicanonical_h2_problem(
        system, rhf, hamiltonian
    )
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=[0, 1],
        virtual_orbitals=list(range(2, rhf.nmo)),
        orbital_energies=eps,
        max_rank=case["rank"],
    )
    solver = getattr(forte2, f"solve_sparse_mrdsrg{case['rank']}")
    result = solver(
        hamiltonian,
        vacuum,
        rhf.nmo,
        excitations,
        flow_param=case["flow_param"],
        max_cumulant=max(3, min(case["rank"], 4)),
        e_tol=E_TOL,
        r_tol=R_TOL,
        maxiter=MAX_ITER,
        max_commutators=20,
        do_diis=True,
        diis_start=3,
        diis_nvec=8,
    )
    return {
        "status": "ok" if result.converged else "not_converged",
        "converged": result.converged,
        "energy": result.energy,
        "scalar_energy": result.scalar_energy,
        "reference_energy": reference_energy,
        "fci": fci,
        "rhf_energy": float(rhf.E),
        "nmo": rhf.nmo,
        "n_amplitudes": len(result.amplitudes),
        "iterations": result.iterations,
        "solve_s": result.seconds,
        "max_cumulant": result.max_cumulant,
        "history_tail": [vars(item) for item in result.history[-5:]],
    }


def run_case(case: dict) -> dict:
    started = time.perf_counter()
    try:
        result = run_sr(case) if case["method"] == "sr" else run_mr(case)
    except BaseException as exc:
        result = {
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    result.update(case)
    result["wall_s"] = time.perf_counter() - started
    result["completed_at"] = now_s()
    return result


def _case_target(queue, case: dict) -> None:
    queue.put(run_case(case))


def run_with_timeout(case: dict, timeout_s: float) -> dict:
    context = mp.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_case_target, args=(queue, case))
    started = time.perf_counter()
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            **case,
            "status": "timeout",
            "timeout_s": timeout_s,
            "wall_s": time.perf_counter() - started,
            "completed_at": now_s(),
        }
    if queue.empty():
        return {
            **case,
            "status": "no_result",
            "exitcode": process.exitcode,
            "wall_s": time.perf_counter() - started,
            "completed_at": now_s(),
        }
    return queue.get()


def worker(root: Path, worker_id: str, timeout_s: float) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    for case in manifest["cases"]:
        result_path = root / "cases" / f"{case['id']}.json"
        lock_path = root / "locks" / case["id"]
        if result_path.exists():
            continue
        try:
            lock_path.mkdir()
        except FileExistsError:
            continue
        try:
            if result_path.exists():
                continue
            print(f"START {now_s()} worker={worker_id} {case['id']}", flush=True)
            result = run_with_timeout(case, timeout_s)
            atomic_json(result_path, result)
            print(
                f"DONE {now_s()} worker={worker_id} {case['id']} "
                f"status={result['status']} wall_s={result['wall_s']:.2f}",
                flush=True,
            )
        finally:
            lock_path.rmdir()


def reap(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    results = []
    for case in manifest["cases"]:
        path = root / "cases" / f"{case['id']}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    payload = {
        "metadata": manifest["metadata"] | {"reaped_at": now_s()},
        "summary": {
            "total": len(manifest["cases"]),
            "completed": len(results),
            "pending": len(manifest["cases"]) - len(results),
            "status_counts": counts,
        },
        "cases": results,
    }
    atomic_json(root / "results.json", payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sow", "worker", "reap"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-id", default=str(os.getpid()))
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--reset-locks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sow":
        sow(args.root, args.reset_locks)
    elif args.mode == "worker":
        worker(args.root, args.worker_id, args.timeout)
    else:
        reap(args.root)


if __name__ == "__main__":
    main()
