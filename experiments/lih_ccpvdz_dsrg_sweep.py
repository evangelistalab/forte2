"""Restartable LiH/cc-pVDZ SR- and MR-LDSRG(n) benchmark sweep."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

import forte2
from experiments import dsrg_hchain_benchmark as sr
from experiments.lih_cumulant_wick_benchmark import cas22_reference
from forte2 import CI, RHF, State, System
from forte2.base_classes import CIParams, DavidsonLiuParams
from forte2.helpers import DIIS, logger
from forte2.lib.det import Determinant
from forte2.lib.sparse_ops import (
    CumulantReference,
    SparseOperator,
    SparseState,
    overlap,
    sparse_operator_hamiltonian,
)

BASIS = "cc-pVDZ"
EQUILIBRIUM_BOND_LENGTH = 1.60
BOND_RATIOS = (1, 2, 3)
RANKS = (2, 3, 4)
FLOW_PARAM = 5.0
E_TOL = 1.0e-10
R_TOL = 1.0e-5
MAX_ITER = 80
MAX_COMMUTATORS = 20
SCREEN = 1.0e-12
DIIS_START = 3
DIIS_NVEC = 8
DIIS_MIN = 3


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def case_id(case: dict) -> str:
    ratio = case["bond_ratio"]
    if case["method"] == "fci":
        return f"fci_R{ratio}re"
    return f"{case['method']}_R{ratio}re_n{case['rank']}"


def make_manifest() -> dict:
    cases = []
    for ratio in BOND_RATIOS:
        case = {
            "method": "fci",
            "bond_ratio": ratio,
            "bond_length_angstrom": ratio * EQUILIBRIUM_BOND_LENGTH,
        }
        case["id"] = case_id(case)
        cases.append(case)
    for ratio in BOND_RATIOS:
        for rank in RANKS:
            for method in ("sr_normal", "mr_normal"):
                case = {
                    "method": method,
                    "rank": rank,
                    "bond_ratio": ratio,
                    "bond_length_angstrom": ratio * EQUILIBRIUM_BOND_LENGTH,
                }
                case["id"] = case_id(case)
                cases.append(case)
        if ratio == 1:
            for rank in RANKS:
                case = {
                    "method": "sr_bare",
                    "rank": rank,
                    "bond_ratio": ratio,
                    "bond_length_angstrom": ratio * EQUILIBRIUM_BOND_LENGTH,
                }
                case["id"] = case_id(case)
                cases.append(case)
    return {
        "metadata": {
            "created_at": now_s(),
            "system": "LiH",
            "basis": BASIS,
            "equilibrium_bond_length_angstrom": EQUILIBRIUM_BOND_LENGTH,
            "bond_ratios": list(BOND_RATIOS),
            "flow_param": FLOW_PARAM,
            "e_tol": E_TOL,
            "r_tol": R_TOL,
            "max_iter": MAX_ITER,
            "max_commutators": MAX_COMMUTATORS,
            "screen_thresh": SCREEN,
            "diis": {
                "enabled": True,
                "start": DIIS_START,
                "nvec": DIIS_NVEC,
                "min": DIIS_MIN,
            },
            "sr_normal_truncation": "determinant-normal-ordered many-body rank",
            "sr_bare_truncation": "bare physical operator-string many-body rank",
            "mr_reference": "frozen-core CAS(2e,2o), generalized-Fock semicanonical",
            "mr_backend": "sparse cumulant-truncated GNO",
            "mr_max_cumulant": 3,
            "mr_include_four_body_cumulant": False,
        },
        "cases": cases,
    }


def sow(root: Path, reset_locks: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(exist_ok=True)
    (root / "locks").mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        atomic_json(manifest_path, make_manifest())
    if reset_locks:
        for lock in (root / "locks").iterdir():
            if lock.is_dir():
                lock.rmdir()
    print(f"SOWED {len(json.loads(manifest_path.read_text())['cases'])} cases")


def build_lih_problem(bond_length: float, basis: str = BASIS) -> dict:
    logger.set_verbosity_level(0)
    xyz = f"Li 0.0 0.0 0.0\nH 0.0 0.0 {bond_length:.12f}"
    with contextlib.redirect_stdout(io.StringIO()):
        system = System(
            xyz=xyz,
            basis_set=basis,
            minao_basis_set=None,
            cholesky_tei=True,
            cholesky_tol=1.0e-12,
        )
        rhf = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system)
        rhf.run()

    coefficients = rhf.C[0]
    hcore = np.einsum(
        "pq,pi,qj->ij", system.ints_hcore(), coefficients, coefficients, optimize=True
    )
    eri = system.fock_builder.two_electron_integrals_block(coefficients)
    hamiltonian = sparse_operator_hamiltonian(system.nuclear_repulsion, hcore, eri)
    reference = Determinant("2" * rhf.na + "0" * (rhf.nmo - rhf.na))

    mr_vacuum, _, _ = cas22_reference(hamiltonian, rhf.nmo)
    cumulants = CumulantReference(mr_vacuum, rhf.nmo, max_cumulant=3)
    gamma = np.array(
        [
            [
                sum(cumulants.gamma(p, spin, q, spin).real for spin in (True, False))
                for q in range(rhf.nmo)
            ]
            for p in range(rhf.nmo)
        ]
    )
    fock = hcore + np.einsum("rs,prqs->pq", gamma, eri, optimize=True)
    fock -= 0.5 * np.einsum("rs,prsq->pq", gamma, eri, optimize=True)

    rotation = np.zeros_like(fock)
    for orbital_slice in (slice(0, 1), slice(1, 3), slice(3, rhf.nmo)):
        _, vectors = np.linalg.eigh(fock[orbital_slice, orbital_slice])
        rotation[orbital_slice, orbital_slice] = vectors
    mr_orbital_energies = np.diag(rotation.T @ fock @ rotation)
    mr_hcore = rotation.T @ hcore @ rotation
    mr_eri = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        rotation,
        rotation,
        eri,
        rotation,
        rotation,
        optimize=True,
    )
    mr_hamiltonian = sparse_operator_hamiltonian(
        system.nuclear_repulsion, mr_hcore, mr_eri
    )
    mr_vacuum, mr_model_space, mr_reference_energy = cas22_reference(
        mr_hamiltonian, rhf.nmo
    )

    return {
        "system": system,
        "rhf": rhf,
        "hamiltonian": hamiltonian,
        "orbital_energies": np.asarray(rhf.eps[0]),
        "reference": reference,
        "mr_hamiltonian": mr_hamiltonian,
        "mr_orbital_energies": mr_orbital_energies,
        "mr_vacuum": mr_vacuum,
        "mr_model_space": mr_model_space,
        "mr_reference_energy": mr_reference_energy,
    }


def fci_energy(problem: dict) -> dict:
    rhf = problem["rhf"]
    ndet = math.comb(rhf.nmo, rhf.na) * math.comb(rhf.nmo, rhf.nb)
    algorithm = "exact" if ndet <= 10000 else "hz"
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solver = CI(
            states=State(nel=rhf.na + rhf.nb, multiplicity=1, ms=0.0),
            active_orbitals=list(range(rhf.nmo)),
            ci_params=CIParams(ci_algorithm=algorithm),
            davidson_liu_params=DavidsonLiuParams(
                e_tol=E_TOL, r_tol=R_TOL, maxiter=200
            ),
        )(rhf)
        solver.run()
    return {
        "energy": float(solver.E[0]),
        "seconds": time.perf_counter() - started,
        "algorithm": algorithm,
        "ndet": ndet,
        "converged": bool(solver.get_convergence_status()[0]),
    }


def sparse_body_rank(term) -> int:
    count = term.count()
    if count % 2:
        raise ValueError(f"particle-number-conserving string required: {term}")
    return count // 2


def truncate_bare_rank(operator, max_rank: int, screen_thresh: float = SCREEN):
    result = SparseOperator()
    for term, coefficient in operator:
        if abs(coefficient) > screen_thresh and sparse_body_rank(term) <= max_rank:
            result.add(term, coefficient)
    return result


def make_bare_cluster_operator(excitations, amplitudes):
    result = SparseOperator()
    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) > SCREEN:
            result += excitation["op"] * complex(amplitude)
    return result


def bch_hbar_bare(
    hamiltonian,
    cluster,
    truncation_rank: int,
    max_commutators: int = MAX_COMMUTATORS,
    commutator_threshold: float = SCREEN,
):
    antihermitian = cluster - cluster.adjoint()
    hbar = truncate_bare_rank(hamiltonian, truncation_rank)
    nested = hbar
    norms = []
    for ncomm in range(1, max_commutators + 1):
        nested = truncate_bare_rank(nested.commutator(antihermitian), truncation_rank)
        contribution = nested * (1.0 / math.factorial(ncomm))
        hbar += contribution
        norm = sr.sparse_operator_norm(contribution)
        norms.append(norm)
        if norm < commutator_threshold:
            break
    return truncate_bare_rank(hbar, truncation_rank), norms


def solve_sr_bare(
    hamiltonian,
    reference,
    excitations,
    truncation_rank: int,
    flow_param: float = FLOW_PARAM,
    e_tol: float = E_TOL,
    r_tol: float = R_TOL,
    max_iter: int = MAX_ITER,
    max_commutators: int = MAX_COMMUTATORS,
):
    reference_state = SparseState({reference: 1.0})
    excitation_states = [
        excitation["op"] @ reference_state for excitation in excitations
    ]
    h0_state = truncate_bare_rank(hamiltonian, truncation_rank) @ reference_state
    h0 = np.array(
        [overlap(state, h0_state) for state in excitation_states], dtype=complex
    )
    amplitudes = np.array(
        [
            h0[index] * sr.regularized_denominator(excitation["denom"], flow_param)
            for index, excitation in enumerate(excitations)
        ],
        dtype=complex,
    )
    diis = DIIS(
        diis_start=DIIS_START,
        diis_nvec=DIIS_NVEC,
        diis_min=DIIS_MIN,
        do_diis=True,
    )
    previous_energy = None
    history = []
    started = time.perf_counter()
    for iteration in range(max_iter + 1):
        iter_started = time.perf_counter()
        cluster = make_bare_cluster_operator(excitations, amplitudes)
        hbar, commutator_norms = bch_hbar_bare(
            hamiltonian,
            cluster,
            truncation_rank,
            max_commutators=max_commutators,
        )
        hbar_reference = hbar @ reference_state
        energy = float(overlap(reference_state, hbar_reference).real)
        offdiagonal = np.array(
            [overlap(state, hbar_reference) for state in excitation_states],
            dtype=complex,
        )
        fixed_point = np.array(
            [
                (offdiagonal[index] + excitation["denom"] * amplitudes[index])
                * sr.regularized_denominator(excitation["denom"], flow_param)
                for index, excitation in enumerate(excitations)
            ],
            dtype=complex,
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
                "n_terms": len(hbar),
                "iter_s": time.perf_counter() - iter_started,
            }
        )
        if (
            previous_energy is not None
            and abs(delta_energy) < e_tol
            and rms_update < r_tol
        ):
            return {
                "energy": energy,
                "amplitudes": amplitudes,
                "history": history,
                "seconds": time.perf_counter() - started,
                "converged": True,
            }
        amplitudes = diis.update(fixed_point, update)
        previous_energy = energy
    return {
        "energy": energy,
        "amplitudes": amplitudes,
        "history": history,
        "seconds": time.perf_counter() - started,
        "converged": False,
    }


def sr_excitations(problem: dict, rank: int):
    rhf = problem["rhf"]
    return sr.enumerate_spin_conserving_excitations(
        rhf.nmo,
        rhf.na,
        problem["orbital_energies"],
        problem["reference"],
        max_excitation_rank=rank,
    )


def solve_sr_normal(
    problem: dict,
    rank: int,
    max_iter: int = MAX_ITER,
    max_commutators: int = MAX_COMMUTATORS,
) -> dict:
    excitations = sr_excitations(problem, rank)
    energy, amplitudes, history, seconds, converged = sr.solve_sparse_dsrg(
        ham=problem["hamiltonian"],
        ref=problem["reference"],
        excitations=excitations,
        truncation_rank=rank,
        flow_param=FLOW_PARAM,
        e_tol=E_TOL,
        r_tol=R_TOL,
        max_iter=max_iter,
        use_diis=True,
        diis_start=DIIS_START,
        diis_nvec=DIIS_NVEC,
        diis_min=DIIS_MIN,
        max_comm=max_commutators,
    )
    return {
        "energy": float(energy),
        "amplitudes": amplitudes,
        "history": history,
        "seconds": seconds,
        "converged": converged,
        "n_amplitudes": len(excitations),
    }


def solve_sr_bare_problem(
    problem: dict,
    rank: int,
    max_iter: int = MAX_ITER,
    max_commutators: int = MAX_COMMUTATORS,
) -> dict:
    excitations = sr_excitations(problem, rank)
    result = solve_sr_bare(
        problem["hamiltonian"],
        problem["reference"],
        excitations,
        rank,
        max_iter=max_iter,
        max_commutators=max_commutators,
    )
    result["n_amplitudes"] = len(excitations)
    return result


def solve_mr_normal(problem: dict, rank: int) -> dict:
    rhf = problem["rhf"]
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[0],
        active_orbitals=[1, 2],
        virtual_orbitals=list(range(3, rhf.nmo)),
        orbital_energies=problem["mr_orbital_energies"],
        max_rank=rank,
    )
    solver = getattr(forte2, f"solve_sparse_mrdsrg{rank}")
    result = solver(
        problem["mr_hamiltonian"],
        problem["mr_vacuum"],
        rhf.nmo,
        excitations,
        flow_param=FLOW_PARAM,
        max_cumulant=3,
        include_four_body_cumulant=False,
        gno_backend="sparse",
        e_tol=E_TOL,
        r_tol=R_TOL,
        maxiter=MAX_ITER,
        max_commutators=MAX_COMMUTATORS,
        do_diis=True,
        diis_start=DIIS_START,
        diis_nvec=DIIS_NVEC,
    )
    return {
        "energy": float(result.energy),
        "scalar_energy": float(result.scalar_energy),
        "amplitudes": result.amplitudes,
        "history": [vars(item) for item in result.history],
        "seconds": result.seconds,
        "converged": result.converged,
        "n_amplitudes": len(excitations),
        "reference_energy": problem["mr_reference_energy"],
    }


def summarize_solver_result(result: dict) -> dict:
    amplitudes = result.pop("amplitudes")
    history = result["history"]
    result.update(
        {
            "iterations": len(history),
            "max_abs_amplitude": (
                float(max(abs(value) for value in amplitudes))
                if len(amplitudes)
                else 0.0
            ),
        }
    )
    return result


def run_case(case: dict) -> dict:
    started = time.perf_counter()
    try:
        problem = build_lih_problem(case["bond_length_angstrom"])
        setup_s = time.perf_counter() - started
        method = case["method"]
        if method == "fci":
            result = {"status": "ok", "fci": fci_energy(problem)}
        elif method == "sr_normal":
            result = summarize_solver_result(solve_sr_normal(problem, case["rank"]))
            result["status"] = "ok" if result["converged"] else "not_converged"
        elif method == "sr_bare":
            result = summarize_solver_result(
                solve_sr_bare_problem(problem, case["rank"])
            )
            result["status"] = "ok" if result["converged"] else "not_converged"
        elif method == "mr_normal":
            result = summarize_solver_result(solve_mr_normal(problem, case["rank"]))
            result["status"] = "ok" if result["converged"] else "not_converged"
        else:
            raise ValueError(f"unknown method {method}")
        result.update(
            {
                "rhf_energy": float(problem["rhf"].E),
                "nmo": problem["rhf"].nmo,
                "setup_s": setup_s,
            }
        )
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


def worker(
    root: Path,
    worker_id: str,
    timeout_s: float,
    methods: set[str] | None = None,
    ranks: set[int] | None = None,
    ratios: set[int] | None = None,
) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    for case in manifest["cases"]:
        if methods and case["method"] not in methods:
            continue
        if ranks and "rank" in case and case["rank"] not in ranks:
            continue
        if ratios and case["bond_ratio"] not in ratios:
            continue
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
    fci_by_ratio = {}
    for case in manifest["cases"]:
        path = root / "cases" / f"{case['id']}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        results.append(result)
        if result["method"] == "fci" and result["status"] == "ok":
            fci_by_ratio[result["bond_ratio"]] = result["fci"]["energy"]
    for result in results:
        if result["method"] == "fci" or "energy" not in result:
            continue
        fci = fci_by_ratio.get(result["bond_ratio"])
        if fci is not None:
            result["fci_energy"] = fci
            result["energy_error"] = result["energy"] - fci

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
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("fci", "sr_normal", "sr_bare", "mr_normal"),
    )
    parser.add_argument("--ranks", nargs="+", type=int, choices=RANKS)
    parser.add_argument("--ratios", nargs="+", type=int, choices=BOND_RATIOS)
    parser.add_argument("--reset-locks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sow":
        sow(args.root, args.reset_locks)
    elif args.mode == "worker":
        worker(
            args.root,
            args.worker_id,
            args.timeout,
            methods=set(args.methods) if args.methods else None,
            ranks=set(args.ranks) if args.ranks else None,
            ratios=set(args.ratios) if args.ratios else None,
        )
    else:
        reap(args.root)


if __name__ == "__main__":
    main()
