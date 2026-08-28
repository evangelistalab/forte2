"""Restartable H4/6-31G SR/MR bare/normal SR-LDSRG(n) sweep."""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "forte2-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "forte2-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forte2
import forte2.dsrg.sparse_mrdsrg2 as mr_impl
from experiments import dsrg_hchain_benchmark as sr
from experiments import lih_ccpvdz_dsrg_sweep as existing
from forte2.helpers import DIIS, logger
from forte2.lib.det import Determinant
from forte2.lib.sparse_ops import (
    CumulantReference,
    GeneralizedNormalOrderedSparseOperator,
    SparseState,
    generalized_normal_order,
    overlap,
    sparse_operator_hamiltonian,
)

BASIS = "6-31G"
BOND_LENGTHS = (0.75, 1.50, 2.25)
RANKS = (2, 3, 4)
METHODS = ("sr_normal", "sr_bare", "mr_normal", "mr_bare")
FLOW_EXPONENTS = tuple(-3.0 + 0.25 * index for index in range(21))
SCREEN = 1.0e-12
MAX_CUMULANT = 3
E_TOL = 1.0e-10
R_TOL = 1.0e-5
MAX_ITER = 80
MAX_COMMUTATORS = 20
DIIS_START = 3
DIIS_NVEC = 8


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=complex))
    temporary.replace(path)


def exponent_tag(exponent: float) -> str:
    sign = "m" if exponent < 0.0 else "p"
    return f"{sign}{abs(exponent):.2f}".replace(".", "p")


def bond_tag(bond_length: float) -> str:
    return f"{bond_length:.2f}".replace(".", "p")


def case_id(case: dict) -> str:
    bond = bond_tag(case["bond_length"])
    if case["method"] == "fci":
        return f"fci_R{bond}"
    return (
        f"{case['method']}_R{bond}_n{case['rank']}_"
        f"s10e{exponent_tag(case['flow_exponent'])}"
    )


def make_manifest() -> dict:
    cases = []
    for bond_length in BOND_LENGTHS:
        case = {"method": "fci", "bond_length": bond_length}
        case["id"] = case_id(case)
        cases.append(case)

    # Rank-four cases are listed first so independent workers share the expensive tail.
    for rank in reversed(RANKS):
        for method in METHODS:
            for bond_length in BOND_LENGTHS:
                for exponent in FLOW_EXPONENTS:
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
            "system": "linear H4",
            "geometry_note": "bond_length is the nearest-neighbor H-H spacing",
            "basis": BASIS,
            "basis_note": (
                "The earlier H4 bare-vs-normal tutorial used STO-3G; 6-31G is "
                "used here as a practical split-valence double-zeta basis."
            ),
            "bond_lengths_angstrom": list(BOND_LENGTHS),
            "ranks": list(RANKS),
            "methods": list(METHODS),
            "flow_exponents": list(FLOW_EXPONENTS),
            "flow_params": [10.0**exponent for exponent in FLOW_EXPONENTS],
            "screen_thresh": SCREEN,
            "max_cumulant": MAX_CUMULANT,
            "include_four_body_cumulant": False,
            "e_tol": E_TOL,
            "r_tol": R_TOL,
            "max_iter": MAX_ITER,
            "max_commutators": MAX_COMMUTATORS,
            "mr_reference": (
                "CAS(4e,4o) in the four lowest RHF orbitals, semicanonicalized "
                "within active and virtual blocks"
            ),
            "mr_normal_definition": (
                "GNO amplitudes and cumulant-Wick BCH commutators truncated by GNO rank"
            ),
            "mr_bare_definition": (
                "The same GNO amplitude parametrization, converted to physical strings for "
                "bare-rank BCH commutators; GNO is used only at iteration boundaries to "
                "extract scalar and residual coefficients."
            ),
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
    manifest = json.loads(manifest_path.read_text())
    print(f"SOWED {len(manifest['cases'])} cases")


def determinant_label(alpha: tuple[int, ...], beta: tuple[int, ...], norb: int) -> str:
    alpha_set = set(alpha)
    beta_set = set(beta)
    occupations = []
    for orbital in range(norb):
        if orbital in alpha_set and orbital in beta_set:
            occupations.append("2")
        elif orbital in alpha_set:
            occupations.append("a")
        elif orbital in beta_set:
            occupations.append("b")
        else:
            occupations.append("0")
    return "".join(occupations)


def cas44_model_space(norb: int) -> tuple[Determinant, ...]:
    return tuple(
        Determinant(determinant_label(alpha, beta, norb))
        for alpha in itertools.combinations(range(4), 2)
        for beta in itertools.combinations(range(4), 2)
    )


def cas44_reference(hamiltonian, norb: int):
    if norb < 4:
        raise ValueError("CAS(4,4) requires at least four spatial orbitals")
    model_space = cas44_model_space(norb)
    matrix = np.zeros((len(model_space), len(model_space)), dtype=complex)
    for column, ket in enumerate(model_space):
        hket = hamiltonian @ SparseState({ket: 1.0})
        for row, bra in enumerate(model_space):
            matrix[row, column] = hket[bra]
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    vacuum = SparseState(
        {
            determinant: coefficient
            for determinant, coefficient in zip(model_space, eigenvectors[:, 0])
            if abs(coefficient) > 1.0e-14
        }
    )
    return vacuum, model_space, float(eigenvalues[0].real)


def semicanonical_mr_problem(system, rhf, hcore, eri):
    norb = rhf.nmo
    hamiltonian = sparse_operator_hamiltonian(system.nuclear_repulsion, hcore, eri)
    vacuum, _, _ = cas44_reference(hamiltonian, norb)
    cumulants = CumulantReference(
        vacuum, norb, max_cumulant=MAX_CUMULANT, screen_thresh=SCREEN
    )
    gamma = np.array(
        [
            [
                sum(cumulants.gamma(p, alpha, q, alpha).real for alpha in (True, False))
                for q in range(norb)
            ]
            for p in range(norb)
        ]
    )
    fock = hcore + np.einsum("rs,prqs->pq", gamma, eri, optimize=True)
    fock -= 0.5 * np.einsum("rs,prsq->pq", gamma, eri, optimize=True)

    rotation = np.zeros_like(fock)
    for orbital_slice in (slice(0, 4), slice(4, norb)):
        if orbital_slice.start == orbital_slice.stop:
            continue
        _, vectors = np.linalg.eigh(fock[orbital_slice, orbital_slice])
        rotation[orbital_slice, orbital_slice] = vectors
    orbital_energies = np.diag(rotation.T @ fock @ rotation)
    rotated_hcore = rotation.T @ hcore @ rotation
    rotated_eri = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        rotation,
        rotation,
        eri,
        rotation,
        rotation,
        optimize=True,
    )
    rotated_hamiltonian = sparse_operator_hamiltonian(
        system.nuclear_repulsion, rotated_hcore, rotated_eri
    )
    vacuum, model_space, reference_energy = cas44_reference(rotated_hamiltonian, norb)
    return {
        "hamiltonian": rotated_hamiltonian,
        "vacuum": vacuum,
        "model_space": model_space,
        "reference_energy": reference_energy,
        "orbital_energies": orbital_energies,
    }


def build_problem(bond_length: float) -> dict:
    logger.set_verbosity_level(0)
    sr.BASIS = BASIS
    with contextlib.redirect_stdout(io.StringIO()):
        system, rhf, hamiltonian, orbital_energies = (
            sr.build_linear_h_sparse_hamiltonian(4, bond_length)
        )
    coefficients = rhf.C[0]
    hcore = np.einsum(
        "pq,pi,qj->ij",
        system.ints_hcore(),
        coefficients,
        coefficients,
        optimize=True,
    )
    eri = system.fock_builder.two_electron_integrals_block(coefficients)
    reference = Determinant("2" * rhf.na + "0" * (rhf.nmo - rhf.na))
    return {
        "system": system,
        "rhf": rhf,
        "hamiltonian": hamiltonian,
        "orbital_energies": orbital_energies,
        "reference": reference,
        "mr": semicanonical_mr_problem(system, rhf, hcore, eri),
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


def mr_excitations(problem: dict, rank: int):
    norb = problem["rhf"].nmo
    return forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=list(range(4)),
        virtual_orbitals=list(range(4, norb)),
        orbital_energies=problem["mr"]["orbital_energies"],
        max_rank=rank,
    )


def summarize_amplitudes(result: dict) -> dict:
    amplitudes = result.pop("amplitudes")
    history = result.pop("history")
    result.update(
        {
            "iterations": len(history),
            "history_tail": history[-8:],
            "max_abs_amplitude": (
                float(max(abs(value) for value in amplitudes))
                if len(amplitudes)
                else 0.0
            ),
        }
    )
    return result


def solve_sr_normal(problem: dict, rank: int, flow_param: float) -> dict:
    excitations = sr_excitations(problem, rank)
    energy, amplitudes, history, seconds, converged = sr.solve_sparse_dsrg(
        ham=problem["hamiltonian"],
        ref=problem["reference"],
        excitations=excitations,
        truncation_rank=rank,
        flow_param=flow_param,
        e_tol=E_TOL,
        r_tol=R_TOL,
        max_iter=MAX_ITER,
        use_diis=True,
        diis_start=DIIS_START,
        diis_nvec=DIIS_NVEC,
        diis_min=3,
        max_comm=MAX_COMMUTATORS,
    )
    return summarize_amplitudes(
        {
            "energy": float(energy),
            "amplitudes": amplitudes,
            "history": history,
            "seconds": seconds,
            "converged": converged,
            "n_amplitudes": len(excitations),
        }
    )


def solve_sr_bare(problem: dict, rank: int, flow_param: float) -> dict:
    excitations = sr_excitations(problem, rank)
    result = existing.solve_sr_bare(
        problem["hamiltonian"],
        problem["reference"],
        excitations,
        rank,
        flow_param=flow_param,
        e_tol=E_TOL,
        r_tol=R_TOL,
        max_iter=MAX_ITER,
        max_commutators=MAX_COMMUTATORS,
    )
    result["n_amplitudes"] = len(excitations)
    result["seconds"] = result.pop("seconds")
    return summarize_amplitudes(result)


def run_mr_normal(
    problem: dict,
    rank: int,
    flow_param: float,
    initial_amplitudes=None,
    iteration_callback=None,
    damping: float = 1.0,
):
    excitations = mr_excitations(problem, rank)
    solver = getattr(forte2, f"solve_sparse_mrdsrg{rank}")
    return solver(
        problem["mr"]["hamiltonian"],
        problem["mr"]["vacuum"],
        problem["rhf"].nmo,
        excitations,
        flow_param=flow_param,
        max_cumulant=MAX_CUMULANT,
        include_four_body_cumulant=False,
        gno_backend="cumulant",
        screen_thresh=SCREEN,
        commutator_threshold=SCREEN,
        e_tol=E_TOL,
        r_tol=R_TOL,
        maxiter=MAX_ITER,
        max_commutators=MAX_COMMUTATORS,
        do_diis=True,
        diis_start=DIIS_START,
        diis_nvec=DIIS_NVEC,
        initial_amplitudes=initial_amplitudes,
        iteration_callback=iteration_callback,
        damping=damping,
    )


def summarize_mr_normal(result, reference_energy: float) -> dict:
    return {
        "energy": float(result.energy),
        "scalar_energy": float(result.scalar_energy),
        "converged": result.converged,
        "iterations": result.iterations,
        "history_tail": [vars(item) for item in result.history[-8:]],
        "seconds": result.seconds,
        "n_amplitudes": len(result.amplitudes),
        "max_abs_amplitude": (
            float(max(abs(value) for value in result.amplitudes))
            if len(result.amplitudes)
            else 0.0
        ),
        "reference_energy": reference_energy,
        "gno_backend": result.gno_backend,
    }


def solve_mr_normal(problem: dict, rank: int, flow_param: float) -> dict:
    result = run_mr_normal(problem, rank, flow_param)
    return summarize_mr_normal(result, problem["mr"]["reference_energy"])


def make_mr_cluster(problem: dict, excitations, amplitudes):
    cluster = GeneralizedNormalOrderedSparseOperator(
        problem["mr"]["vacuum"], problem["rhf"].nmo, MAX_CUMULANT
    )
    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) > SCREEN:
            cluster.add(excitation.sqop, complex(amplitude))
    return cluster


def solve_mr_bare(
    problem: dict,
    rank: int,
    flow_param: float,
    initial_amplitudes=None,
    iteration_callback=None,
    summarize: bool = True,
) -> dict:
    hamiltonian = problem["mr"]["hamiltonian"]
    vacuum = problem["mr"]["vacuum"]
    norb = problem["rhf"].nmo
    excitations = mr_excitations(problem, rank)
    h0 = generalized_normal_order(
        hamiltonian,
        vacuum,
        norb,
        max_cumulant=MAX_CUMULANT,
        screen_thresh=SCREEN,
        max_rank=rank,
    )
    excitations = mr_impl._with_normal_ordered_denominators(excitations, h0, norb)
    if initial_amplitudes is None:
        amplitudes = np.array(
            [
                h0.coefficient(excitation.sqop)
                * mr_impl.regularized_denominator(excitation.denominator, flow_param)
                for excitation in excitations
            ],
            dtype=complex,
        )
    else:
        amplitudes = np.asarray(initial_amplitudes, dtype=complex).copy()
        if amplitudes.shape != (len(excitations),):
            raise ValueError("initial_amplitudes has the wrong shape")
    identity = mr_impl.identity_sqop()
    diis = DIIS(
        diis_start=DIIS_START,
        diis_nvec=DIIS_NVEC,
        diis_min=3,
        do_diis=True,
    )
    history = []
    previous_energy = None
    started = time.perf_counter()
    converged = False
    for iteration in range(MAX_ITER + 1):
        iter_started = time.perf_counter()
        cluster = make_mr_cluster(problem, excitations, amplitudes)
        bare_cluster = cluster.to_sparse_operator(SCREEN)
        hbar_bare, commutator_norms = existing.bch_hbar_bare(
            hamiltonian,
            bare_cluster,
            rank,
            max_commutators=MAX_COMMUTATORS,
            commutator_threshold=SCREEN,
        )
        hbar = generalized_normal_order(
            hbar_bare,
            vacuum,
            norb,
            max_cumulant=MAX_CUMULANT,
            screen_thresh=SCREEN,
            max_rank=rank,
        )
        energy = float(hbar.coefficient(identity).real)
        offdiagonal = np.array(
            [hbar.coefficient(excitation.sqop) for excitation in excitations],
            dtype=complex,
        )
        fixed_point = np.array(
            [
                (offdiagonal[index] + excitation.denominator * amplitudes[index])
                * mr_impl.regularized_denominator(excitation.denominator, flow_param)
                for index, excitation in enumerate(excitations)
            ],
            dtype=complex,
        )
        update = fixed_point - amplitudes
        rms_update = float(np.linalg.norm(update) / math.sqrt(len(update)))
        delta_energy = 0.0 if previous_energy is None else energy - previous_energy
        history.append(
            {
                "iteration": iteration,
                "energy": energy,
                "delta_energy": delta_energy,
                "rms_update": rms_update,
                "ncomm": len(commutator_norms),
                "iter_s": time.perf_counter() - iter_started,
                "bare_terms": len(hbar_bare),
                "gno_terms": len(hbar),
            }
        )
        if iteration_callback is not None:
            iteration_callback(history[-1])
        if (
            previous_energy is not None
            and abs(delta_energy) < E_TOL
            and rms_update < R_TOL
        ):
            converged = True
            break
        amplitudes = diis.update(fixed_point, update)
        previous_energy = energy

    result = {
        "energy": energy,
        "scalar_energy": energy,
        "amplitudes": amplitudes,
        "history": history,
        "seconds": time.perf_counter() - started,
        "converged": converged,
        "n_amplitudes": len(excitations),
        "reference_energy": problem["mr"]["reference_energy"],
    }
    return summarize_amplitudes(result) if summarize else result


SOLVERS = {
    "sr_normal": solve_sr_normal,
    "sr_bare": solve_sr_bare,
    "mr_normal": solve_mr_normal,
    "mr_bare": solve_mr_bare,
}


def run_case(case: dict) -> dict:
    started = time.perf_counter()
    try:
        if sr.SCREEN != SCREEN or existing.SCREEN != SCREEN:
            raise RuntimeError(
                f"operator screens are normal={sr.SCREEN}, bare={existing.SCREEN}; "
                f"expected {SCREEN}"
            )
        problem = build_problem(case["bond_length"])
        setup_s = time.perf_counter() - started
        if case["method"] == "fci":
            with contextlib.redirect_stdout(io.StringIO()):
                result = {
                    "fci": sr.forte2_fci_energy(problem["system"], problem["rhf"], 4),
                    "status": "ok",
                }
        else:
            result = SOLVERS[case["method"]](problem, case["rank"], case["flow_param"])
            result["status"] = "ok" if result["converged"] else "not_converged"
        result.update(
            {
                "rhf_energy": float(problem["rhf"].E),
                "cas_reference_energy": problem["mr"]["reference_energy"],
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

    def stop_process() -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()

    try:
        process.join(timeout_s)
    except BaseException:
        stop_process()
        raise
    if process.is_alive():
        stop_process()
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


def selected(case: dict, args) -> bool:
    if args.methods and case["method"] not in args.methods:
        return False
    if args.ranks and case.get("rank") not in args.ranks:
        return False
    if args.bond_lengths and case["bond_length"] not in args.bond_lengths:
        return False
    if args.flow_exponents and case.get("flow_exponent") not in args.flow_exponents:
        return False
    return True


def should_run_result(result_path: Path, rerun_statuses: set[str]) -> bool:
    if not result_path.exists():
        return True
    if not rerun_statuses:
        return False
    previous = json.loads(result_path.read_text())
    return previous.get("status") in rerun_statuses


def worker(root: Path, args) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    rerun_statuses = set(args.rerun_statuses or ())
    for case in manifest["cases"]:
        if not selected(case, args):
            continue
        result_path = root / "cases" / f"{case['id']}.json"
        lock_path = root / "locks" / case["id"]
        if not should_run_result(result_path, rerun_statuses):
            continue
        try:
            lock_path.mkdir()
        except FileExistsError:
            continue
        try:
            if not should_run_result(result_path, rerun_statuses):
                continue
            print(f"START {now_s()} worker={args.worker_id} {case['id']}", flush=True)
            result = run_with_timeout(case, args.timeout)
            atomic_json(result_path, result)
            print(
                f"DONE {now_s()} worker={args.worker_id} {case['id']} "
                f"status={result['status']} wall_s={result['wall_s']:.2f}",
                flush=True,
            )
        finally:
            lock_path.rmdir()


def continue_sr_normal(root: Path, args) -> None:
    if not args.ranks or len(args.ranks) != 1:
        raise ValueError("Continuation requires exactly one --ranks value")
    if not args.bond_lengths or len(args.bond_lengths) != 1:
        raise ValueError("Continuation requires exactly one --bond-lengths value")
    if not args.flow_exponents:
        raise ValueError("Continuation requires --flow-exponents")

    rank = args.ranks[0]
    bond_length = args.bond_lengths[0]
    exponents = sorted(set(args.flow_exponents))
    manifest = json.loads((root / "manifest.json").read_text())
    cases = {
        (case.get("rank"), case["bond_length"], case.get("flow_exponent")): case
        for case in manifest["cases"]
        if case["method"] == "sr_normal"
    }

    setup_started = time.perf_counter()
    problem = build_problem(bond_length)
    excitations = sr_excitations(problem, rank)
    setup_s = time.perf_counter() - setup_started
    amplitudes = None

    for exponent in exponents:
        case = cases.get((rank, bond_length, exponent))
        if case is None:
            raise ValueError(
                f"No manifest case for rank={rank}, R={bond_length}, exponent={exponent}"
            )

        print(
            f"START {now_s()} continuation {case['id']} "
            f"guess={'first-order' if amplitudes is None else 'previous-s'}",
            flush=True,
        )

        def report_iteration(item: dict) -> None:
            print(
                f"ITER {case['id']} {item['iteration']:02d} "
                f"E={item['energy']:.12f} dE={item['delta_energy']:+.3e} "
                f"|dt|={item['rms_update']:.3e} ncomm={item['ncomm']} "
                f"iter_s={item['iter_s']:.2f}",
                flush=True,
            )

        started = time.perf_counter()
        energy, next_amplitudes, history, seconds, converged = sr.solve_sparse_dsrg(
            ham=problem["hamiltonian"],
            ref=problem["reference"],
            excitations=excitations,
            truncation_rank=rank,
            flow_param=case["flow_param"],
            e_tol=E_TOL,
            r_tol=R_TOL,
            max_iter=MAX_ITER,
            use_diis=True,
            diis_start=DIIS_START,
            diis_nvec=DIIS_NVEC,
            diis_min=3,
            max_comm=MAX_COMMUTATORS,
            initial_amplitudes=amplitudes,
            iteration_callback=report_iteration,
        )
        result = summarize_amplitudes(
            {
                "energy": float(energy),
                "amplitudes": next_amplitudes,
                "history": history,
                "seconds": seconds,
                "converged": converged,
                "n_amplitudes": len(excitations),
            }
        )
        result.update(
            {
                **case,
                "status": "ok" if converged else "not_converged",
                "rhf_energy": float(problem["rhf"].E),
                "cas_reference_energy": problem["mr"]["reference_energy"],
                "nmo": problem["rhf"].nmo,
                "setup_s": setup_s,
                "wall_s": time.perf_counter() - started,
                "completed_at": now_s(),
                "initial_guess": (
                    "first_order" if amplitudes is None else "previous_flow_parameter"
                ),
            }
        )
        atomic_json(root / "cases" / f"{case['id']}.json", result)
        print(
            f"DONE {now_s()} continuation {case['id']} "
            f"status={result['status']} wall_s={result['wall_s']:.2f}",
            flush=True,
        )
        if not converged:
            print(
                "STOP continuation because the latest point did not converge",
                flush=True,
            )
            break
        amplitudes = next_amplitudes


def continue_rank(root: Path, args) -> None:
    if not args.ranks or len(args.ranks) != 1:
        raise ValueError("Rank continuation requires exactly one --ranks value")
    rank = args.ranks[0]
    if args.mode == "continue-rank2" and rank != 2:
        raise ValueError("continue-rank2 requires --ranks 2")
    if not args.bond_lengths or len(args.bond_lengths) != 1:
        raise ValueError("Rank continuation requires one --bond-lengths value")
    if not args.flow_exponents:
        raise ValueError("Rank continuation requires --flow-exponents")

    if args.methods:
        methods = args.methods
    elif args.mode == "continue-rank2":
        methods = ["sr_normal", "mr_normal", "mr_bare"]
    else:
        methods = list(METHODS)
    supported = set(METHODS)
    if not set(methods) <= supported:
        raise ValueError(f"Rank continuation supports {sorted(supported)}")

    bond_length = args.bond_lengths[0]
    target_exponents = sorted(set(args.flow_exponents))
    step = args.continuation_step
    if step <= 0.0:
        raise ValueError("--continuation-step must be positive")
    exponents = [target_exponents[0]]
    for target in target_exponents[1:]:
        current = exponents[-1]
        while current + step < target - 1.0e-10:
            current = round(current + step, 10)
            exponents.append(current)
        exponents.append(target)
    target_keys = {round(exponent, 10) for exponent in target_exponents}
    manifest = json.loads((root / "manifest.json").read_text())
    cases = {
        (case["method"], case["bond_length"], case.get("flow_exponent")): case
        for case in manifest["cases"]
        if case.get("rank") == rank and case["method"] in supported
    }

    setup_started = time.perf_counter()
    problem = build_problem(bond_length)
    setup_s = time.perf_counter() - setup_started
    sr_rank_excitations = sr_excitations(problem, rank)

    for method in methods:
        amplitudes = None
        amplitude_history = []
        for exponent in exponents:
            is_target = round(exponent, 10) in target_keys
            case = cases.get((method, bond_length, exponent)) if is_target else None
            if is_target and case is None:
                raise ValueError(
                    f"No manifest case for method={method}, R={bond_length}, "
                    f"exponent={exponent}"
                )
            if case is None:
                case = {
                    "id": (
                        f"internal_{method}_R{bond_tag(bond_length)}_n{rank}_"
                        f"x{exponent:+.10f}"
                    ),
                    "method": method,
                    "bond_length": bond_length,
                    "rank": rank,
                    "flow_exponent": exponent,
                    "flow_param": 10.0**exponent,
                }

            amplitude_path = root / "amplitudes" / f"{case['id']}.npy"
            if amplitude_path.exists():
                amplitudes = np.load(amplitude_path)
                amplitude_history.append((exponent, amplitudes))
                amplitude_history = amplitude_history[-2:]
                print(
                    f"REUSE {now_s()} continuation {case['id']} "
                    "amplitudes=checkpoint",
                    flush=True,
                )
                continue

            initial_amplitudes = amplitudes
            initial_guess = (
                "first_order" if amplitudes is None else "previous_flow_parameter"
            )
            if args.extrapolate_amplitudes and len(amplitude_history) == 2:
                (x0, a0), (x1, a1) = amplitude_history
                if abs(x1 - x0) > 1.0e-12:
                    scale = (exponent - x1) / (x1 - x0)
                    initial_amplitudes = a1 + scale * (a1 - a0)
                    initial_guess = "linear_exponent_extrapolation"

            print(
                f"START {now_s()} continuation {case['id']} " f"guess={initial_guess}",
                flush=True,
            )

            def report_iteration(item) -> None:
                values = item if isinstance(item, dict) else vars(item)
                print(
                    f"ITER {case['id']} {values['iteration']:02d} "
                    f"E={values['energy']:.12f} "
                    f"dE={values['delta_energy']:+.3e} "
                    f"|dt|={values['rms_update']:.3e} "
                    f"ncomm={values['ncomm']} iter_s={values['iter_s']:.2f}",
                    flush=True,
                )

            started = time.perf_counter()
            if method == "sr_normal":
                energy, next_amplitudes, history, seconds, converged = (
                    sr.solve_sparse_dsrg(
                        ham=problem["hamiltonian"],
                        ref=problem["reference"],
                        excitations=sr_rank_excitations,
                        truncation_rank=rank,
                        flow_param=case["flow_param"],
                        e_tol=E_TOL,
                        r_tol=R_TOL,
                        max_iter=MAX_ITER,
                        use_diis=True,
                        diis_start=DIIS_START,
                        diis_nvec=DIIS_NVEC,
                        diis_min=3,
                        max_comm=MAX_COMMUTATORS,
                        initial_amplitudes=initial_amplitudes,
                        iteration_callback=report_iteration,
                        damping=args.damping,
                    )
                )
                result = summarize_amplitudes(
                    {
                        "energy": float(energy),
                        "amplitudes": next_amplitudes,
                        "history": history,
                        "seconds": seconds,
                        "converged": converged,
                        "n_amplitudes": len(sr_rank_excitations),
                    }
                )
            elif method == "sr_bare":
                raw_result = existing.solve_sr_bare(
                    problem["hamiltonian"],
                    problem["reference"],
                    sr_rank_excitations,
                    rank,
                    flow_param=case["flow_param"],
                    e_tol=E_TOL,
                    r_tol=R_TOL,
                    max_iter=MAX_ITER,
                    max_commutators=MAX_COMMUTATORS,
                    initial_amplitudes=initial_amplitudes,
                    iteration_callback=report_iteration,
                    damping=args.damping,
                )
                next_amplitudes = raw_result["amplitudes"]
                converged = raw_result["converged"]
                raw_result["n_amplitudes"] = len(sr_rank_excitations)
                result = summarize_amplitudes(raw_result)
            elif method == "mr_normal":
                raw_result = run_mr_normal(
                    problem,
                    rank,
                    case["flow_param"],
                    initial_amplitudes=initial_amplitudes,
                    iteration_callback=report_iteration,
                    damping=args.damping,
                )
                next_amplitudes = raw_result.amplitudes
                converged = raw_result.converged
                result = summarize_mr_normal(
                    raw_result, problem["mr"]["reference_energy"]
                )
            else:
                raw_result = solve_mr_bare(
                    problem,
                    rank,
                    case["flow_param"],
                    initial_amplitudes=initial_amplitudes,
                    iteration_callback=report_iteration,
                    summarize=False,
                )
                next_amplitudes = raw_result["amplitudes"]
                converged = raw_result["converged"]
                result = summarize_amplitudes(raw_result)

            result.update(
                {
                    **case,
                    "status": "ok" if converged else "not_converged",
                    "rhf_energy": float(problem["rhf"].E),
                    "cas_reference_energy": problem["mr"]["reference_energy"],
                    "nmo": problem["rhf"].nmo,
                    "setup_s": setup_s,
                    "wall_s": time.perf_counter() - started,
                    "completed_at": now_s(),
                    "initial_guess": initial_guess,
                }
            )
            if is_target:
                atomic_json(root / "cases" / f"{case['id']}.json", result)
            if converged:
                atomic_npy(amplitude_path, next_amplitudes)
            print(
                f"DONE {now_s()} continuation {case['id']} "
                f"status={result['status']} wall_s={result['wall_s']:.2f}",
                flush=True,
            )
            if not converged:
                print(
                    f"STOP continuation for {method}: point did not converge",
                    flush=True,
                )
                break
            amplitudes = next_amplitudes
            amplitude_history.append((exponent, amplitudes))
            amplitude_history = amplitude_history[-2:]


def reap(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    results = []
    fci_by_bond = {}
    for case in manifest["cases"]:
        path = root / "cases" / f"{case['id']}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        results.append(result)
        if result["method"] == "fci" and result["status"] == "ok":
            fci_by_bond[result["bond_length"]] = result["fci"]["energy"]
    for result in results:
        if result["method"] == "fci" or result.get("energy") is None:
            continue
        fci_energy = fci_by_bond.get(result["bond_length"])
        if fci_energy is not None:
            result["fci_energy"] = fci_energy
            result["energy_error"] = result["energy"] - fci_energy

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    coverage = []
    for method in METHODS:
        for rank in RANKS:
            for bond_length in BOND_LENGTHS:
                matching = [
                    result
                    for result in results
                    if result["method"] == method
                    and result.get("rank") == rank
                    and result["bond_length"] == bond_length
                ]
                coverage.append(
                    {
                        "method": method,
                        "rank": rank,
                        "bond_length": bond_length,
                        "converged": sum(
                            result["status"] == "ok" for result in matching
                        ),
                        "attempted": len(matching),
                        "requested": len(FLOW_EXPONENTS),
                    }
                )
    payload = {
        "metadata": manifest["metadata"] | {"reaped_at": now_s()},
        "summary": {
            "total": len(manifest["cases"]),
            "completed": len(results),
            "pending": len(manifest["cases"]) - len(results),
            "status_counts": counts,
            "coverage": coverage,
        },
        "cases": results,
    }
    atomic_json(root / "results.json", payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return payload


def plot_quantity(
    successful: list[dict], output: Path, quantity: str, ylabel: str
) -> None:
    if quantity not in {"energy", "energy_error"}:
        raise ValueError(f"Unsupported plot quantity: {quantity}")

    is_error = quantity == "energy_error"
    colors = {2: "#0072B2", 3: "#D55E00", 4: "#009E73"}
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.5), sharex=True)
    for row_index, reference in enumerate(("sr", "mr")):
        for column, bond_length in enumerate(BOND_LENGTHS):
            axis = axes[row_index, column]
            fci_energy = next(
                case["fci_energy"]
                for case in successful
                if case["bond_length"] == bond_length
            )
            if not is_error:
                axis.axhline(
                    fci_energy,
                    color="#444444",
                    linestyle=":",
                    linewidth=1.1,
                    label="FCI",
                )
            for rank in RANKS:
                for representation, linestyle, marker in (
                    ("normal", "-", "o"),
                    ("bare", "--", "s"),
                ):
                    method = f"{reference}_{representation}"
                    rows = sorted(
                        (
                            case
                            for case in successful
                            if case["method"] == method
                            and case["bond_length"] == bond_length
                            and case["rank"] == rank
                        ),
                        key=lambda case: case["flow_param"],
                    )
                    if not rows:
                        continue
                    values = [case[quantity] for case in rows]
                    if is_error:
                        values = [max(abs(value) * 1000.0, 1.0e-10) for value in values]
                    axis.plot(
                        [case["flow_param"] for case in rows],
                        values,
                        color=colors[rank],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.35,
                        markersize=3,
                        label=f"{representation.title()} n={rank}",
                    )
            axis.set_xscale("log")
            if is_error:
                axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.2)
            axis.set_title(f"{reference.upper()}, R = {bond_length:.2f} Angstrom")
            if row_index == 1:
                axis.set_xlabel(r"Flow parameter $s$ / Eh$^{-2}$")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(
        "Linear H4/6-31G DSRG(n): available converged checkpoints",
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"WROTE {output}")


def plot(root: Path, output: Path) -> None:
    results = json.loads((root / "results.json").read_text())
    method_cases = [case for case in results["cases"] if case["method"] in METHODS]
    successful = [case for case in method_cases if case["status"] == "ok"]
    if not successful:
        raise RuntimeError("No converged DSRG cases are available")

    energy_output = output.with_name(f"{output.stem}_energy{output.suffix}")
    error_output = output.with_name(f"{output.stem}_error{output.suffix}")
    plot_quantity(
        successful,
        energy_output,
        "energy",
        r"Total energy / E$_\mathrm{h}$",
    )
    plot_quantity(
        successful,
        error_output,
        "energy_error",
        "Absolute FCI energy error (mEh)",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "sow",
            "worker",
            "continue-sr-normal",
            "continue-rank2",
            "continue-rank",
            "reap",
            "plot",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-id", default=str(os.getpid()))
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--reset-locks", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("fci", *METHODS))
    parser.add_argument("--ranks", nargs="+", type=int, choices=RANKS)
    parser.add_argument("--bond-lengths", nargs="+", type=float)
    parser.add_argument("--flow-exponents", nargs="+", type=float)
    parser.add_argument(
        "--continuation-step",
        type=float,
        default=0.25,
        help="Maximum exponent step between continuation solves.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Fixed-point damping used by continuation solves.",
    )
    parser.add_argument(
        "--extrapolate-amplitudes",
        action="store_true",
        help="Linearly extrapolate amplitudes from the two preceding exponents.",
    )
    parser.add_argument(
        "--rerun-statuses",
        nargs="+",
        choices=("timeout", "not_converged", "error", "no_result"),
        help="Replace existing checkpoints having one of these statuses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/h4_dz_dsrg_sweep.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sow":
        sow(args.root, args.reset_locks)
    elif args.mode == "worker":
        worker(args.root, args)
    elif args.mode == "continue-sr-normal":
        continue_sr_normal(args.root, args)
    elif args.mode in {"continue-rank2", "continue-rank"}:
        continue_rank(args.root, args)
    elif args.mode == "reap":
        reap(args.root)
    else:
        plot(args.root, args.output)


if __name__ == "__main__":
    main()
