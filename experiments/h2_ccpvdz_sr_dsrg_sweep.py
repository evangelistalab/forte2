"""Sweep the SR-LDSRG(n) flow parameter for H2/cc-pVDZ."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "forte2-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "forte2-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from experiments import dsrg_hchain_benchmark as benchmark
except ModuleNotFoundError:
    import dsrg_hchain_benchmark as benchmark


BASIS = "cc-pVDZ"
BOND_LENGTH = 0.75
DEFAULT_EXPONENTS = tuple(-4.0 + 0.5 * index for index in range(15))
DEFAULT_RANKS = (2, 3, 4)


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def initialize_results(args) -> dict:
    return {
        "metadata": {
            "created_at": now_s(),
            "updated_at": now_s(),
            "system": "H2",
            "bond_length_angstrom": args.bond_length,
            "geometry": f"H 0 0 0; H 0 0 {args.bond_length} Angstrom",
            "basis": BASIS,
            "method": "determinant-normal-ordered SR-LDSRG(n)",
            "ranks": list(args.ranks),
            "flow_exponents": list(args.exponents),
            "flow_values": [10.0**exponent for exponent in args.exponents],
            "e_tol": args.e_tol,
            "r_tol": args.r_tol,
            "max_iter": args.max_iter,
            "case_timeout_s": args.case_timeout,
            "diis": {
                "enabled": not args.no_diis,
                "start": args.diis_start,
                "nvec": args.diis_nvec,
                "min": args.diis_min,
            },
        },
        "cases": [],
    }


def load_results(path: Path, args) -> dict:
    if not path.exists():
        return initialize_results(args)
    results = json.loads(path.read_text())
    metadata = results.get("metadata", {})
    if metadata.get("basis") != BASIS or metadata.get("system") != "H2":
        raise ValueError(f"{path} contains results for a different system")
    stored_bond_length = metadata.get("bond_length_angstrom", BOND_LENGTH)
    if abs(stored_bond_length - args.bond_length) > 1.0e-12:
        raise ValueError(
            f"{path} contains results for bond length {stored_bond_length} Angstrom"
        )
    return results


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def case_key(row: dict) -> tuple[int | None, float | None]:
    return row.get("rank"), row.get("flow_exponent")


def run_sweep(args) -> dict:
    output = Path(args.output)
    results = load_results(output, args)
    completed = {
        case_key(row)
        for row in results["cases"]
        if row.get("status") == "ok" and not args.rerun
    }

    benchmark.BASIS = BASIS
    for rank in args.ranks:
        for exponent in args.exponents:
            key = (rank, exponent)
            if key in completed:
                print(f"SKIP SR-LDSRG({rank}) N={exponent:+.1f}", flush=True)
                continue

            flow_param = 10.0**exponent
            print(
                f"START {now_s()} SR-LDSRG({rank}) "
                f"N={exponent:+.1f} s={flow_param:.8g}",
                flush=True,
            )
            case_args = (
                2,
                args.bond_length,
                rank,
                flow_param,
                args.e_tol,
                args.r_tol,
                args.max_iter,
                not args.no_diis,
                args.diis_start,
                args.diis_nvec,
                args.diis_min,
            )
            row = benchmark.run_case_with_timeout(case_args, args.case_timeout)
            row["flow_exponent"] = exponent
            row["completed_at"] = now_s()
            results["cases"] = [
                old_row for old_row in results["cases"] if case_key(old_row) != key
            ]
            results["cases"].append(row)
            results["metadata"]["updated_at"] = now_s()
            save_results(output, results)

            energy = row.get("dsrg_energy")
            energy_text = "n/a" if energy is None else f"{energy:.15f}"
            print(
                f"DONE status={row['status']} E={energy_text} "
                f"iterations={row.get('iterations', 'n/a')} "
                f"solve_s={row.get('solve_s', row.get('elapsed_s', 0.0)):.2f}",
                flush=True,
            )
    return results


def plot_results(results: dict, path: Path) -> None:
    successful = [row for row in results["cases"] if row.get("status") == "ok"]
    if not successful:
        raise RuntimeError("No converged results are available to plot")

    references = successful[0]
    rhf_energy = references["rhf_energy"]
    fci_energy = references["fci"]["energy"]
    colors = {2: "#0072B2", 3: "#D55E00", 4: "#009E73"}

    figure, (energy_axis, error_axis) = plt.subplots(
        2, 1, figsize=(7.2, 7.2), sharex=True
    )
    for rank in sorted({row["rank"] for row in successful}):
        rows = sorted(
            (row for row in successful if row["rank"] == rank),
            key=lambda row: row["flow_param"],
        )
        flow_values = [row["flow_param"] for row in rows]
        energies = [row["dsrg_energy"] for row in rows]
        label = f"SR-LDSRG({rank})"
        color = colors.get(rank)
        energy_axis.plot(
            flow_values, energies, marker="o", markersize=4, label=label, color=color
        )
        error_axis.plot(
            flow_values,
            [abs(energy - fci_energy) * 1000.0 for energy in energies],
            marker="o",
            markersize=4,
            label=label,
            color=color,
        )

    energy_axis.axhline(fci_energy, color="#222222", linestyle="--", label="FCI")
    energy_axis.axhline(rhf_energy, color="#777777", linestyle=":", label="RHF")
    energy_axis.set_ylabel("Energy / Eh")
    error_axis.set_ylabel("|Energy - FCI| / mEh")
    error_axis.set_xlabel("Flow parameter s / Eh$^{-2}$")
    error_axis.set_xscale("log")
    error_axis.set_yscale("log")
    energy_axis.grid(True, alpha=0.25)
    error_axis.grid(True, alpha=0.25)
    energy_axis.legend(ncol=2)
    bond_length = results["metadata"].get("bond_length_angstrom", BOND_LENGTH)
    figure.suptitle(
        rf"H$_2$/cc-pVDZ, $R_{{\mathrm{{H-H}}}}={bond_length:.2f}$ Angstrom",
        y=0.98,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="experiments/h2_ccpvdz_sr_dsrg_sweep_results.json"
    )
    parser.add_argument("--plot", default="experiments/h2_ccpvdz_sr_dsrg_sweep.png")
    parser.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    parser.add_argument("--exponents", nargs="+", type=float, default=DEFAULT_EXPONENTS)
    parser.add_argument("--bond-length", type=float, default=BOND_LENGTH)
    parser.add_argument("--e-tol", type=float, default=1.0e-10)
    parser.add_argument("--r-tol", type=float, default=1.0e-5)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--case-timeout", type=float, default=600.0)
    parser.add_argument("--no-diis", action="store_true")
    parser.add_argument("--diis-start", type=int, default=3)
    parser.add_argument("--diis-nvec", type=int, default=8)
    parser.add_argument("--diis-min", type=int, default=3)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        results = load_results(Path(args.output), args)
    else:
        results = run_sweep(args)
    plot_results(results, Path(args.plot))
    print(f"WROTE {args.output}", flush=True)
    print(f"WROTE {args.plot}", flush=True)


if __name__ == "__main__":
    main()
