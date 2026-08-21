#!/usr/bin/env python3
"""Time S1 SA-CASSCF gradients for all-trans polyenes in cc-pVTZ.

The complete pi space of C(2p) orbitals is selected with AVAS. Two singlet
roots are optimized with equal weights, and the timed gradient is for absolute
root 1 (S1).

Run one or more members from the repository root, for example::

    FORTE_NUM_THREADS_OVERRIDE=4 OMP_NUM_THREADS=4 \
        python benchmarks/sa_casscf_polyene_gradient_timings.py ethene butadiene
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from math import atan2, cos, pi, sin
from time import perf_counter

from forte2 import AVAS, CISolver, MCOptimizer, RHF, State, System, set_verbosity_level
from forte2.lib.cpp_helpers import get_num_threads


@dataclass(frozen=True)
class Polyene:
    name: str
    double_bonds: int

    @property
    def active_space(self) -> str:
        electrons = 2 * self.double_bonds
        return f"CAS({electrons}e,{electrons}o)"


POLYENES = {
    polyene.name: polyene
    for polyene in (
        Polyene("ethene", 1),
        Polyene("butadiene", 2),
        Polyene("hexatriene", 3),
        Polyene("octatetraene", 4),
        Polyene("decapentaene", 5),
    )
}


def all_trans_geometry(double_bonds: int) -> str:
    """Return an idealized planar C_(2n)H_(2n+2) geometry in Angstrom."""
    ncarbon = 2 * double_bonds
    carbon_positions = [(0.0, 0.0)]
    for bond in range(ncarbon - 1):
        length = 1.34 if bond % 2 == 0 else 1.46
        angle = pi / 6 if bond % 2 == 0 else -pi / 6
        x, y = carbon_positions[-1]
        carbon_positions.append((x + length * cos(angle), y + length * sin(angle)))

    hydrogen_positions = []
    ch_length = 1.09
    for carbon, (x, y) in enumerate(carbon_positions):
        if carbon == 0:
            neighbor_angle = pi / 6
            hydrogen_angles = (neighbor_angle + 2 * pi / 3, neighbor_angle - 2 * pi / 3)
        elif carbon == ncarbon - 1:
            previous_angle = pi / 6 if (carbon - 1) % 2 == 0 else -pi / 6
            neighbor_angle = previous_angle + pi
            hydrogen_angles = (neighbor_angle + 2 * pi / 3, neighbor_angle - 2 * pi / 3)
        else:
            previous_angle = pi / 6 if (carbon - 1) % 2 == 0 else -pi / 6
            next_angle = pi / 6 if carbon % 2 == 0 else -pi / 6
            back = (cos(previous_angle + pi), sin(previous_angle + pi))
            forward = (cos(next_angle), sin(next_angle))
            hydrogen_angles = (atan2(-(back[1] + forward[1]), -(back[0] + forward[0])),)
        for angle in hydrogen_angles:
            hydrogen_positions.append(
                (x + ch_length * cos(angle), y + ch_length * sin(angle))
            )

    lines = [f"C {x:.10f} {y:.10f} 0.0" for x, y in carbon_positions]
    lines.extend(f"H {x:.10f} {y:.10f} 0.0" for x, y in hydrogen_positions)
    return "\n".join(lines)


def build_reference(polyene: Polyene):
    """Build a two-root, equal-weight SA-CASSCF reference."""
    system = System(
        xyz=all_trans_geometry(polyene.double_bonds),
        basis_set="cc-pVTZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="angstrom",
    )
    rhf = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8, maxiter=100)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C", "H"]],
        selection_method="separate",
        num_active_docc=polyene.double_bonds,
        num_active_uocc=polyene.double_bonds,
    )(rhf)
    avas.run()

    npi = polyene.double_bonds
    ncore = avas.mo_space.ncore
    active_pi = list(range(ncore, ncore + 2 * npi))
    state = State(system=system, multiplicity=1, ms=0.0)
    ci_solver = CISolver(
        state,
        core_orbitals=list(range(ncore)),
        active_orbitals=active_pi,
        nroots=2,
        weights=[0.5, 0.5],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1.0e-8,
        g_tol=1.0e-6,
        maxiter=100,
        final_orbitals="original",
    )(avas)
    mc.run()
    return mc


def time_polyene(polyene: Polyene):
    """Return setup and first-excited-singlet gradient wall times."""
    print(f"[{polyene.name}] building {polyene.active_space} reference", flush=True)
    start = perf_counter()
    mc = build_reference(polyene)
    reference_seconds = perf_counter() - start
    sub_solver = mc.ci_solver.sub_solvers[0]
    print(
        f"[{polyene.name}] reference ready in {reference_seconds:.3f} s; "
        f"timing S1 gradient",
        flush=True,
    )
    start = perf_counter()
    gradient = mc.gradient(root=1)
    gradient_seconds = perf_counter() - start
    return {
        "name": polyene.name,
        "double_bonds": polyene.double_bonds,
        "space": polyene.active_space,
        "nbf": mc.system.nbf,
        "naux": mc.system.naux,
        "ncsf": sub_solver.basis_size,
        "nrot": mc.orb_opt.nrot,
        "reference_seconds": reference_seconds,
        "gradient_seconds": gradient_seconds,
        "gradient_norm": float((gradient**2).sum() ** 0.5),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "systems",
        choices=tuple(POLYENES),
        nargs="*",
        help="Polyenes to benchmark (default: all, smallest to largest).",
    )
    parser.add_argument(
        "--max-gradient-seconds",
        type=float,
        default=600.0,
        help="Stop the series after a gradient exceeds this time (default: 600 s).",
    )
    args = parser.parse_args()
    systems = args.systems or tuple(POLYENES)

    set_verbosity_level(0)
    results = []
    for name in systems:
        result = time_polyene(POLYENES[name])
        results.append(result)
        print(
            f"[{name}] gradient completed in {result['gradient_seconds']:.3f} s",
            flush=True,
        )
        if result["gradient_seconds"] > args.max_gradient_seconds:
            print(
                f"Stopping: gradient exceeded {args.max_gradient_seconds:.1f} s.",
                flush=True,
            )
            break

    print(
        "\nbasis: cc-pVTZ/cc-pVTZ-JKFIT; SA weights: [0.5, 0.5]; "
        f"target: root 1 (S1); threads: {get_num_threads()}"
    )
    print(
        f"{'system':<16} {'C=C':>4} {'space':<13} {'nbf':>5} {'naux':>5} "
        f"{'nCSF':>8} {'nrot':>6} {'reference / s':>14} {'gradient / s':>13}"
    )
    print("-" * 96)
    for result in results:
        print(
            f"{result['name']:<16} {result['double_bonds']:4d} "
            f"{result['space']:<13} {result['nbf']:5d} {result['naux']:5d} "
            f"{result['ncsf']:8d} {result['nrot']:6d} "
            f"{result['reference_seconds']:14.3f} "
            f"{result['gradient_seconds']:13.3f}"
        )


if __name__ == "__main__":
    main()
