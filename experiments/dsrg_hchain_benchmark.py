import argparse
import contextlib
import io
import itertools
import json
import math
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

from forte2 import CI, RHF, State, System
from forte2.base_classes import CIParams, DavidsonLiuParams
from forte2.helpers import DIIS, logger
from forte2.lib.det import Determinant
from forte2.lib.sparse_ops import (
    NormalOrderedSparseOperator,
    normal_order,
    sparse_operator,
    sparse_operator_hamiltonian,
)

SCREEN = 1.0e-12
BASIS = "sto-3g"


def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def determinant_count(nmo, na, nb):
    return math.comb(nmo, na) * math.comb(nmo, nb)


def load_results(path):
    if path.exists():
        return json.loads(path.read_text())
    return {
        "metadata": {
            "created_at": now_s(),
            "basis": BASIS,
            "screen": SCREEN,
            "geometry": "linear H_n chain along z",
        },
        "cases": [],
    }


def save_results(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_linear_h_sparse_hamiltonian(natoms, spacing):
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

    C = rhf.C[0]
    hcore_mo = np.einsum("pq,pi,qj->ij", system.ints_hcore(), C, C, optimize=True)
    eri_mo = system.fock_builder.two_electron_integrals_block(C)
    ham = sparse_operator_hamiltonian(system.nuclear_repulsion, hcore_mo, eri_mo)
    eps = np.array(rhf.eps[0])
    return system, rhf, ham, eps


def forte2_fci_energy(system, rhf, natoms):
    ndet = determinant_count(rhf.nmo, rhf.na, rhf.nb)
    algorithm = "exact" if ndet <= 10000 else "hz"
    params = CIParams(ci_algorithm=algorithm)
    davidson = DavidsonLiuParams(e_tol=1.0e-10, r_tol=1.0e-5, maxiter=200)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        ci = CI(
            states=State(nel=natoms, multiplicity=1, ms=0.0),
            active_orbitals=list(range(rhf.nmo)),
            ci_params=params,
            davidson_liu_params=davidson,
        )(rhf)
        ci.run()
    return {
        "energy": float(ci.E[0]),
        "seconds": time.perf_counter() - t0,
        "algorithm": algorithm,
        "converged": bool(ci.get_convergence_status()[0]),
        "ndet": ndet,
    }


def normal_key_and_phase(spop, ref):
    no_op = normal_order(spop, ref, SCREEN)
    items = [(term, coeff) for term, coeff in no_op if abs(coeff) > 1.0e-10]
    if len(items) != 1:
        raise RuntimeError(
            "Expected one normal-ordered term, got "
            + str([(term.str(ref), coeff) for term, coeff in items])
        )
    return items[0]


def physical_coeff(no_op, key, phase):
    return no_op.coefficient(key) / phase


def canonical_excitation_string(cre_modes, ann_modes):
    def token(mode, creation):
        orbital, spin = mode
        return f"{orbital}{spin}{'+' if creation else '-'}"

    alpha_cre = sorted([m for m in cre_modes if m[1] == "a"], key=lambda x: x[0])
    beta_cre = sorted([m for m in cre_modes if m[1] == "b"], key=lambda x: x[0])
    beta_ann = sorted(
        [m for m in ann_modes if m[1] == "b"], key=lambda x: x[0], reverse=True
    )
    alpha_ann = sorted(
        [m for m in ann_modes if m[1] == "a"], key=lambda x: x[0], reverse=True
    )

    tokens = [token(m, True) for m in alpha_cre]
    tokens += [token(m, True) for m in beta_cre]
    tokens += [token(m, False) for m in beta_ann]
    tokens += [token(m, False) for m in alpha_ann]
    return "[" + " ".join(tokens) + "]"


def enumerate_spin_conserving_excitations(
    nspatial, nocc, eps, ref, max_excitation_rank
):
    occ = [(i, spin) for i in range(nocc) for spin in ("a", "b")]
    virt = [(a, spin) for a in range(nocc, nspatial) for spin in ("a", "b")]
    highest_rank = min(max_excitation_rank, len(occ), len(virt))
    excitations = []

    for rank in range(1, highest_rank + 1):
        for ann in itertools.combinations(occ, rank):
            ann_spins = sorted(spin for _, spin in ann)
            for cre in itertools.combinations(virt, rank):
                if ann_spins != sorted(spin for _, spin in cre):
                    continue
                label = canonical_excitation_string(cre, ann)
                op = sparse_operator(label, 1.0)
                key, phase = normal_key_and_phase(op, ref)
                denom = sum(eps[i] for i, _ in ann) - sum(eps[a] for a, _ in cre)
                excitations.append(
                    {
                        "rank": rank,
                        "label": label,
                        "op": op,
                        "key": key,
                        "phase": phase,
                        "denom": denom,
                    }
                )
    return excitations


def make_normal_ordered_cluster_operator(excitations, amplitudes, ref):
    T_no = NormalOrderedSparseOperator(ref)
    for ex, amp in zip(excitations, amplitudes):
        if abs(amp) > SCREEN:
            T_no.add(ex["key"], complex(amp) * ex["phase"])
    return T_no


def truncate_sparse_operator(op, ref, truncation_rank):
    return normal_order(op, ref, SCREEN, max_rank=truncation_rank).to_sparse_operator(
        SCREEN
    )


def sparse_operator_norm(op):
    return math.sqrt(sum(abs(coeff) ** 2 for _, coeff in op)) if len(op) else 0.0


def bch_hbar_dsrg(ham, A_no, ref, truncation_rank, max_comm=20, comm_thresh=1.0e-12):
    hbar = normal_order(ham, ref, SCREEN, max_rank=truncation_rank)
    nested = hbar
    commutator_norms = []

    for ncomm in range(1, max_comm + 1):
        nested = nested.commutator(A_no, truncation_rank, SCREEN)
        contribution = nested * (1.0 / math.factorial(ncomm))
        hbar += contribution

        norm = contribution.norm()
        commutator_norms.append(norm)
        if norm < comm_thresh:
            break

    return hbar.truncate(truncation_rank, SCREEN), commutator_norms


def regularized_denominator(denom, flow_param):
    return (1.0 - math.exp(-flow_param * denom * denom)) / denom


def solve_sparse_dsrg(
    ham,
    ref,
    excitations,
    truncation_rank,
    flow_param,
    e_tol,
    r_tol,
    max_iter,
    use_diis,
    diis_start,
    diis_nvec,
    diis_min,
    max_comm=20,
    initial_amplitudes=None,
    iteration_callback=None,
    damping=1.0,
):
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in the interval (0, 1]")
    ham_no = normal_order(ham, ref, SCREEN, max_rank=truncation_rank)
    h0 = np.array(
        [physical_coeff(ham_no, ex["key"], ex["phase"]) for ex in excitations],
        dtype=complex,
    )
    if initial_amplitudes is None:
        amplitudes = np.array(
            [
                h0[k] * regularized_denominator(ex["denom"], flow_param)
                for k, ex in enumerate(excitations)
            ],
            dtype=complex,
        )
    else:
        amplitudes = np.asarray(initial_amplitudes, dtype=complex).copy()
        if amplitudes.shape != h0.shape:
            raise ValueError(
                f"Initial amplitude shape {amplitudes.shape} does not match {h0.shape}"
            )

    diis = DIIS(
        diis_start=diis_start,
        diis_nvec=diis_nvec,
        diis_min=diis_min,
        do_diis=use_diis,
    )
    identity_key, identity_phase = normal_key_and_phase(sparse_operator("[]", 1.0), ref)
    history = []
    previous_energy = None
    solve_t0 = time.perf_counter()

    for iteration in range(max_iter + 1):
        iter_t0 = time.perf_counter()
        T_no = make_normal_ordered_cluster_operator(excitations, amplitudes, ref)
        A_no = T_no - T_no.adjoint(SCREEN)
        hbar_no, commutator_norms = bch_hbar_dsrg(
            ham,
            A_no,
            ref,
            truncation_rank=truncation_rank,
            max_comm=max_comm,
        )
        energy = physical_coeff(hbar_no, identity_key, identity_phase).real
        hbar_offdiag = np.array(
            [physical_coeff(hbar_no, ex["key"], ex["phase"]) for ex in excitations],
            dtype=complex,
        )

        fixed_point = np.array(
            [
                (hbar_offdiag[k] + ex["denom"] * amplitudes[k])
                * regularized_denominator(ex["denom"], flow_param)
                for k, ex in enumerate(excitations)
            ],
            dtype=complex,
        )

        damped_fixed_point = amplitudes + damping * (fixed_point - amplitudes)
        update = damped_fixed_point - amplitudes
        rms_update = float(np.linalg.norm(update))
        delta_energy = 0.0 if previous_energy is None else energy - previous_energy
        next_amplitudes = diis.update(damped_fixed_point, update)
        iter_s = time.perf_counter() - iter_t0
        history.append(
            {
                "iteration": iteration,
                "energy": energy,
                "delta_energy": delta_energy,
                "rms_update": rms_update,
                "ncomm": len(commutator_norms),
                "iter_s": iter_s,
                "elapsed_s": time.perf_counter() - solve_t0,
                "diis_status": getattr(diis, "status", ""),
            }
        )
        if iteration_callback is not None:
            iteration_callback(history[-1])

        if (
            previous_energy is not None
            and abs(delta_energy) < e_tol
            and rms_update < r_tol
        ):
            return energy, amplitudes, history, time.perf_counter() - solve_t0, True

        amplitudes = next_amplitudes
        previous_energy = energy

    return energy, amplitudes, history, time.perf_counter() - solve_t0, False


def run_case(args):
    (
        natoms,
        spacing,
        rank,
        flow_param,
        e_tol,
        r_tol,
        max_iter,
        use_diis,
        diis_start,
        diis_nvec,
        diis_min,
    ) = args
    logger.set_verbosity_level(0)
    t_case = time.perf_counter()
    try:
        t0 = time.perf_counter()
        system, rhf, ham, eps = build_linear_h_sparse_hamiltonian(natoms, spacing)
        setup_s = time.perf_counter() - t0
        ref = Determinant("2" * rhf.na + "0" * (rhf.nmo - rhf.na))
        fci = forte2_fci_energy(system, rhf, natoms)

        t0 = time.perf_counter()
        excitations = enumerate_spin_conserving_excitations(
            rhf.nmo, rhf.na, eps, ref, max_excitation_rank=rank
        )
        enum_s = time.perf_counter() - t0

        energy, amplitudes, history, solve_s, converged = solve_sparse_dsrg(
            ham=ham,
            ref=ref,
            excitations=excitations,
            truncation_rank=rank,
            flow_param=flow_param,
            e_tol=e_tol,
            r_tol=r_tol,
            max_iter=max_iter,
            use_diis=use_diis,
            diis_start=diis_start,
            diis_nvec=diis_nvec,
            diis_min=diis_min,
        )
        return {
            "status": "ok" if converged else "not_converged",
            "natoms": natoms,
            "spacing": spacing,
            "rank": rank,
            "basis": BASIS,
            "flow_param": flow_param,
            "e_tol": e_tol,
            "r_tol": r_tol,
            "diis": {
                "enabled": use_diis,
                "start": diis_start,
                "nvec": diis_nvec,
                "min": diis_min,
                "updates": sum(1 for row in history if row["diis_status"] == "S/E"),
            },
            "nmo": rhf.nmo,
            "na": rhf.na,
            "nb": rhf.nb,
            "reference": ref.str(rhf.nmo),
            "rhf_energy": float(rhf.E),
            "ham_terms": len(ham),
            "setup_s": setup_s,
            "fci": fci,
            "n_amplitudes": len(excitations),
            "enum_s": enum_s,
            "dsrg_energy": float(energy),
            "dsrg_minus_fci": float(energy - fci["energy"]),
            "converged": converged,
            "iterations": len(history),
            "last_ncomm": history[-1]["ncomm"],
            "max_abs_amplitude": (
                float(max(abs(amp) for amp in amplitudes)) if len(amplitudes) else 0.0
            ),
            "solve_s": solve_s,
            "total_s": time.perf_counter() - t_case,
            "history_tail": history[-5:],
        }
    except BaseException as exc:
        return {
            "status": "error",
            "natoms": natoms,
            "spacing": spacing,
            "rank": rank,
            "basis": BASIS,
            "flow_param": flow_param,
            "e_tol": e_tol,
            "r_tol": r_tol,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "total_s": time.perf_counter() - t_case,
        }


def run_case_with_timeout(case_args, timeout_s):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(
        target=lambda queue, payload: queue.put(run_case(payload)), args=(q, case_args)
    )
    t0 = time.perf_counter()
    p.start()
    p.join(timeout_s)
    elapsed = time.perf_counter() - t0
    if p.is_alive():
        p.terminate()
        p.join(10)
        if p.is_alive():
            p.kill()
            p.join()
        natoms, spacing, rank, flow_param, e_tol, r_tol, *_ = case_args
        return {
            "status": "timeout",
            "natoms": natoms,
            "spacing": spacing,
            "rank": rank,
            "basis": BASIS,
            "flow_param": flow_param,
            "e_tol": e_tol,
            "r_tol": r_tol,
            "timeout_s": timeout_s,
            "elapsed_s": elapsed,
        }
    if q.empty():
        natoms, spacing, rank, flow_param, e_tol, r_tol, *_ = case_args
        return {
            "status": "no-result",
            "natoms": natoms,
            "spacing": spacing,
            "rank": rank,
            "basis": BASIS,
            "flow_param": flow_param,
            "e_tol": e_tol,
            "r_tol": r_tol,
            "elapsed_s": elapsed,
            "exitcode": p.exitcode,
        }
    row = q.get()
    row["wall_s"] = elapsed
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/dsrg_hchain_results.json")
    parser.add_argument("--atoms", nargs="+", type=int, default=[8, 10])
    parser.add_argument("--spacings", nargs="+", type=float, default=[0.74, 1.48])
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--flow", type=float, default=5.0)
    parser.add_argument("--e-tol", type=float, default=1.0e-10)
    parser.add_argument("--r-tol", type=float, default=1.0e-5)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=5400.0)
    parser.add_argument("--no-diis", action="store_true")
    parser.add_argument("--diis-start", type=int, default=3)
    parser.add_argument("--diis-nvec", type=int, default=8)
    parser.add_argument("--diis-min", type=int, default=3)
    args = parser.parse_args()

    output = Path(args.output)
    data = load_results(output)
    data["metadata"].update(
        {
            "updated_at": now_s(),
            "flow_param": args.flow,
            "e_tol": args.e_tol,
            "r_tol": args.r_tol,
            "max_iter": args.max_iter,
            "timeout_s": args.timeout,
            "diis_enabled": not args.no_diis,
            "diis_start": args.diis_start,
            "diis_nvec": args.diis_nvec,
            "diis_min": args.diis_min,
        }
    )
    save_results(output, data)

    completed_keys = {
        (
            row.get("natoms"),
            row.get("spacing"),
            row.get("rank"),
            row.get("e_tol"),
            row.get("r_tol"),
            row.get("diis", {}).get("enabled"),
        )
        for row in data["cases"]
        if row.get("status") == "ok"
    }

    for spacing in args.spacings:
        for natoms in args.atoms:
            for rank in args.ranks:
                key = (natoms, spacing, rank, args.e_tol, args.r_tol, not args.no_diis)
                if key in completed_keys:
                    print(
                        f"SKIP completed H{natoms} R={spacing} DSRG({rank})", flush=True
                    )
                    continue
                print(
                    f"START {now_s()} H{natoms} R={spacing} DSRG({rank}) "
                    f"DIIS={not args.no_diis}",
                    flush=True,
                )
                case_args = (
                    natoms,
                    spacing,
                    rank,
                    args.flow,
                    args.e_tol,
                    args.r_tol,
                    args.max_iter,
                    not args.no_diis,
                    args.diis_start,
                    args.diis_nvec,
                    args.diis_min,
                )
                row = run_case_with_timeout(case_args, args.timeout)
                row["completed_at"] = now_s()
                data["cases"].append(row)
                data["metadata"]["updated_at"] = now_s()
                save_results(output, data)
                if row["status"] == "ok":
                    print(
                        f"DONE H{natoms} R={spacing} DSRG({rank}) "
                        f"E={row['dsrg_energy']:.15f} E-FCI={row['dsrg_minus_fci']:.3e} "
                        f"iters={row['iterations']} time={row['solve_s']:.2f}s",
                        flush=True,
                    )
                else:
                    print(f"DONE status={row['status']} row={row}", flush=True)


if __name__ == "__main__":
    main()
