"""Compare sparse and direct-cumulant MR-DSRG backends for LiH/STO-3G."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

import forte2
from forte2.helpers import logger
from forte2.lib.det import Determinant
from forte2.lib.sparse_ops import (
    CumulantReference,
    SparseState,
    overlap,
    sparse_operator_hamiltonian,
)


def determinant_label(occupations: str, norb: int) -> str:
    return occupations + "0" * (norb - len(occupations))


def cas22_reference(hamiltonian, norb: int):
    model_space = tuple(
        Determinant(determinant_label(label, norb))
        for label in ("220", "2ab", "2ba", "202")
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


def build_lih_case(bond_length: float, basis: str):
    """Build LiH and a semicanonical frozen-core CAS(2,2) reference."""
    xyz = f"Li 0.0 0.0 0.0\nH 0.0 0.0 {bond_length:.12f}"
    logger.set_verbosity_level(0)
    with contextlib.redirect_stdout(io.StringIO()):
        system = forte2.System(
            xyz=xyz,
            basis_set=basis,
            minao_basis_set=None,
            cholesky_tei=True,
            cholesky_tol=1.0e-12,
        )
        rhf = forte2.RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system)
        rhf.run()

    coefficients = rhf.C[0]
    hcore_mo = np.einsum(
        "pq,pi,qj->ij",
        system.ints_hcore(),
        coefficients,
        coefficients,
        optimize=True,
    )
    eri_mo = system.fock_builder.two_electron_integrals_block(coefficients)
    hamiltonian = sparse_operator_hamiltonian(
        system.nuclear_repulsion, hcore_mo, eri_mo
    )
    if rhf.nmo < 3:
        raise RuntimeError("LiH CAS(2,2) requires at least three molecular orbitals")
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
    fock = hcore_mo + np.einsum("rs,prqs->pq", gamma, eri_mo, optimize=True)
    fock -= 0.5 * np.einsum("rs,prsq->pq", gamma, eri_mo, optimize=True)

    rotation = np.zeros_like(fock)
    for orbital_slice in (slice(0, 1), slice(1, 3), slice(3, rhf.nmo)):
        _, vectors = np.linalg.eigh(fock[orbital_slice, orbital_slice])
        rotation[orbital_slice, orbital_slice] = vectors
    orbital_energies = np.diag(rotation.T @ fock @ rotation)
    hcore_mo = rotation.T @ hcore_mo @ rotation
    eri_mo = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        rotation,
        rotation,
        eri_mo,
        rotation,
        rotation,
        optimize=True,
    )
    hamiltonian = sparse_operator_hamiltonian(
        system.nuclear_repulsion, hcore_mo, eri_mo
    )
    vacuum, model_space, reference_energy = cas22_reference(hamiltonian, rhf.nmo)
    return {
        "hamiltonian": hamiltonian,
        "vacuum": vacuum,
        "model_space": model_space,
        "norb": rhf.nmo,
        "orbital_energies": orbital_energies,
        "rhf_energy": float(rhf.E),
        "reference_energy": reference_energy,
    }


def run_solver(case: dict, rank: int, backend: str, args) -> dict:
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[0],
        active_orbitals=[1, 2],
        virtual_orbitals=list(range(3, case["norb"])),
        orbital_energies=case["orbital_energies"],
        max_rank=rank,
    )
    solver = getattr(forte2, f"solve_sparse_mrdsrg{rank}")
    start = time.perf_counter()
    result = solver(
        case["hamiltonian"],
        case["vacuum"],
        case["norb"],
        excitations,
        flow_param=args.flow_param,
        max_cumulant=3,
        gno_backend=backend,
        gno_validation_tol=args.validation_tol,
        screen_thresh=args.screen_thresh,
        commutator_threshold=args.commutator_threshold,
        e_tol=args.e_tol,
        r_tol=args.r_tol,
        maxiter=args.maxiter,
        max_commutators=args.max_commutators,
        do_diis=not args.no_diis,
        diis_start=3,
        diis_nvec=8,
    )
    wall_seconds = time.perf_counter() - start
    return {
        "rank": rank,
        "backend": backend,
        "converged": result.converged,
        "energy": result.energy,
        "scalar_energy": result.scalar_energy,
        "iterations": result.iterations,
        "n_excitations": len(excitations),
        "solver_seconds": result.seconds,
        "wall_seconds": wall_seconds,
        "seconds_per_iteration": result.seconds / max(result.iterations, 1),
        "history": [vars(item) for item in result.history],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond-length", type=float, default=1.6)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("sparse", "cumulant", "validate"),
        default=["sparse", "cumulant"],
    )
    parser.add_argument("--flow-param", type=float, default=5.0)
    parser.add_argument("--screen-thresh", type=float, default=1.0e-12)
    parser.add_argument("--commutator-threshold", type=float, default=1.0e-12)
    parser.add_argument("--validation-tol", type=float, default=1.0e-9)
    parser.add_argument("--e-tol", type=float, default=1.0e-10)
    parser.add_argument("--r-tol", type=float, default=1.0e-5)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--max-commutators", type=int, default=20)
    parser.add_argument("--no-diis", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_start = time.perf_counter()
    case = build_lih_case(args.bond_length, args.basis)
    build_seconds = time.perf_counter() - build_start
    payload = {
        "metadata": {
            "system": "LiH",
            "bond_length_angstrom": args.bond_length,
            "basis": args.basis,
            "reference": "frozen-core CAS(2e,2o)",
            "energy": "reference-projected scalar MR-LDSRG energy",
            "core_orbitals": [0],
            "active_orbitals": [1, 2],
            "virtual_orbitals": list(range(3, case["norb"])),
            "norb": case["norb"],
            "max_cumulant": 3,
            "flow_param": args.flow_param,
            "screen_thresh": args.screen_thresh,
            "commutator_threshold": args.commutator_threshold,
            "validation_tol": args.validation_tol,
            "e_tol": args.e_tol,
            "r_tol": args.r_tol,
            "maxiter": args.maxiter,
            "max_commutators": args.max_commutators,
            "diis": not args.no_diis,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "FORTE_NUM_THREADS_OVERRIDE",
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "SLURM_CPUS_PER_TASK",
                )
                if os.environ.get(name) is not None
            },
        },
        "build_seconds": build_seconds,
        "rhf_energy": case["rhf_energy"],
        "reference_energy": case["reference_energy"],
        "results": [],
    }

    for rank in args.ranks:
        for backend in args.backends:
            result = run_solver(case, rank, backend, args)
            payload["results"].append(result)
            print(
                f"MR-DSRG({rank}) {backend:8s}: {result['wall_seconds']:.6f} s, "
                f"{result['iterations']} iterations, E = {result['energy']:.12f}, "
                f"converged = {result['converged']}",
                flush=True,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")

    if args.output is None:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
