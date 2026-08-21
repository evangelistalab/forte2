#!/usr/bin/env python3
"""Compare state-specific and state-averaged CASSCF gradient timings.

The benchmark uses the pi spaces of ethene (CAS(2,2)) and trans-butadiene
(CAS(4,4)) in STO-3G.  It reports the complete RHF+CASSCF setup time and the
median time for one analytic ground-root gradient.  SA-CASSCF uses two
equal-weight singlet roots.  The memory columns compare the actual shared
response workspace with the full transformed DF tensor that the matrix-free
implementation deliberately avoids.

Run from the repository root, for example::

    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python benchmarks/sa_casscf_gradient_timings.py --repeats 7
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Tuple

from forte2 import CISolver, MCOptimizer, RHF, State, System, set_verbosity_level
from forte2.mcopt.mc_optimizer_response import _build_coupled_response_intermediates


@dataclass(frozen=True)
class Polyene:
    name: str
    geometry: str
    core_orbitals: Tuple[int, ...]
    active_orbitals: Tuple[int, ...]
    active_space: str


POLYENES = {
    "ethene": Polyene(
        name="ethene",
        geometry="""
C -0.6695  0.0000  0.0000
C  0.6695  0.0000  0.0000
H -1.2321  0.9237  0.0000
H -1.2321 -0.9237  0.0000
H  1.2321  0.9237  0.0000
H  1.2321 -0.9237  0.0000
""",
        core_orbitals=tuple(range(7)),
        active_orbitals=(7, 8),
        active_space="CAS(2,2)",
    ),
    "butadiene": Polyene(
        name="butadiene",
        geometry="""
C -1.8266  0.0000  0.0000
C -0.6389  0.7025  0.0000
C  0.6389  0.7025  0.0000
C  1.8266  0.0000  0.0000
H -2.7500  0.5400  0.0000
H -1.8300 -1.0800  0.0000
H -0.6400  1.7900  0.0000
H  0.6400  1.7900  0.0000
H  2.7500  0.5400  0.0000
H  1.8300 -1.0800  0.0000
""",
        core_orbitals=tuple(range(13)),
        active_orbitals=(13, 14, 15, 16),
        active_space="CAS(4,4)",
    ),
}


def build_casscf(polyene: Polyene, state_averaged: bool):
    """Build and run the requested CASSCF reference."""
    system = System(
        xyz=polyene.geometry,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="angstrom",
    )
    rhf = RHF(charge=0, e_tol=1.0e-11, d_tol=1.0e-9, maxiter=100)(system)
    solver_options = {"nroots": 2, "weights": [0.5, 0.5]} if state_averaged else {}
    ci_solver = CISolver(
        State(system=system, multiplicity=1, ms=0.0),
        core_orbitals=list(polyene.core_orbitals),
        active_orbitals=list(polyene.active_orbitals),
        **solver_options,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-10,
        g_tol=1.0e-7,
        maxiter=50,
        final_orbitals="original",
    )(rhf)
    mc.run()
    return mc


def time_reference(polyene: Polyene, state_averaged: bool, repeats: int):
    """Time wave-function preparation and repeated root-gradient calls."""
    start = perf_counter()
    mc = build_casscf(polyene, state_averaged)
    wavefunction_seconds = perf_counter() - start

    gradient_seconds = []
    for _ in range(repeats):
        start = perf_counter()
        mc.gradient(root=0 if state_averaged else None)
        gradient_seconds.append(perf_counter() - start)

    response_mib = 0.0
    full_df_mib = 0.0
    if state_averaged:
        intermediates = _build_coupled_response_intermediates(mc.orb_opt)
        unique_arrays = {}
        for block in intermediates:
            for array in block:
                unique_arrays[id(array)] = array
        response_mib = sum(array.nbytes for array in unique_arrays.values()) / 2**20
        full_df_mib = 8.0 * mc.system.naux * mc.mo_space.nmo**2 / 2**20

    return (
        wavefunction_seconds,
        median(gradient_seconds),
        response_mib,
        full_df_mib,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "systems",
        choices=tuple(POLYENES),
        nargs="*",
        help="Polyenes to benchmark (default: ethene and butadiene).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of gradient evaluations used for the median (default: 3).",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    systems = args.systems or tuple(POLYENES)

    set_verbosity_level(0)
    results = []
    for system_name in systems:
        polyene = POLYENES[system_name]
        for state_averaged in (False, True):
            wavefunction, gradient, response_mib, full_df_mib = time_reference(
                polyene, state_averaged, args.repeats
            )
            results.append(
                (
                    system_name,
                    "SA-CASSCF" if state_averaged else "CASSCF",
                    polyene.active_space,
                    wavefunction,
                    gradient,
                    response_mib,
                    full_df_mib,
                )
            )

    print("basis: STO-3G; SA weights: [0.5, 0.5]; target: root 0")
    print(f"gradient time: median of {args.repeats} call(s)")
    print(
        f"{'system':<12} {'method':<11} {'space':<9} "
        f"{'wave function / s':>17} {'gradient / s':>14} "
        f"{'workspace / MiB':>16} {'full B / MiB':>13}"
    )
    print("-" * 100)
    for (
        system_name,
        method,
        active_space,
        wavefunction,
        gradient,
        response_mib,
        full_df_mib,
    ) in results:
        memory = (
            f"{response_mib:16.3f} {full_df_mib:13.3f}"
            if method == "SA-CASSCF"
            else f"{'-':>16} {'-':>13}"
        )
        print(
            f"{system_name:<12} {method:<11} {active_space:<9} "
            f"{wavefunction:17.3f} {gradient:14.3f} {memory}"
        )

    print("\nSA-CASSCF/CASSCF gradient-time ratio")
    for system_name in systems:
        system_results = [row for row in results if row[0] == system_name]
        casscf_gradient = system_results[0][4]
        sa_casscf_gradient = system_results[1][4]
        print(f"{system_name:<12} {sa_casscf_gradient / casscf_gradient:8.2f}x")


if __name__ == "__main__":
    main()
