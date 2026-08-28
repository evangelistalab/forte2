"""Convert Forte2 spinor-modulus cube files to scalar-density cubes.

Forte2's default spinor cube writer stores one real non-negative field for
each spin component:

    a_cube(r) = |phi_alpha(r)|
    b_cube(r) = |phi_beta(r)|

Therefore the scalar density of one two-component spinor is

    rho_p(r) = a_cube(r)^2 + b_cube(r)^2.

For a Kramers pair P = (p, pbar), this script writes the averaged density

    rho_P(r) = [rho_p(r) + rho_pbar(r)] / 2,

which keeps the integral on the same scale as a single normalized spinor.

Examples
--------
One spinor:

    python forte2_density_cube.py spinor \
        h2_ghf_orbs_00_a.cube h2_ghf_orbs_00_b.cube \
        spinor_00_density.cube

One Kramers-pair node:

    python forte2_density_cube.py kpair \
        h2_ghf_orbs_00_a.cube h2_ghf_orbs_00_b.cube \
        h2_ghf_orbs_01_a.cube h2_ghf_orbs_01_b.cube \
        kramers_pair_00_01_density.cube
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class Cube:
    """A single real Gaussian-cube field."""

    natoms: int
    origin: np.ndarray
    grid_counts: tuple[int, int, int]
    shape: tuple[int, int, int]
    axes: np.ndarray
    atom_lines: list[str]
    data: np.ndarray

    @property
    def voxel_volume(self) -> float:
        """Volume of one cube-grid voxel."""

        return float(abs(np.linalg.det(self.axes)))


def _parse_numbers(line: str) -> list[float]:
    """Parse ordinary or Fortran-style floating-point tokens."""

    return [
        float(token.replace("D", "E").replace("d", "e"))
        for token in line.split()
    ]


def read_cube(filename: str | Path) -> Cube:
    """Read a one-field Gaussian cube file."""

    filename = Path(filename)
    lines = filename.read_text().splitlines()

    if len(lines) < 6:
        raise ValueError(f"{filename}: file is too short to be a cube")

    origin_line = _parse_numbers(lines[2])
    if len(origin_line) < 4:
        raise ValueError(f"{filename}: malformed atom-count/origin line")

    natoms = int(origin_line[0])
    if natoms < 0:
        raise ValueError(
            f"{filename}: multi-dataset cube files are not supported; "
            "export each Forte2 component separately"
        )

    origin = np.asarray(origin_line[1:4], dtype=float)

    grid_counts: list[int] = []
    shape: list[int] = []
    axes: list[list[float]] = []

    for line_number in range(3, 6):
        row = _parse_numbers(lines[line_number])
        if len(row) < 4:
            raise ValueError(
                f"{filename}: malformed grid line {line_number + 1}"
            )

        count = int(row[0])
        grid_counts.append(count)
        shape.append(abs(count))
        axes.append(row[1:4])

    atom_end = 6 + natoms
    if len(lines) < atom_end:
        raise ValueError(f"{filename}: incomplete atom block")

    atom_lines = lines[6:atom_end]

    value_tokens = (
        " ".join(lines[atom_end:])
        .replace("D", "E")
        .replace("d", "e")
        .split()
    )
    values = np.asarray([float(token) for token in value_tokens], dtype=float)

    expected_size = int(np.prod(shape))
    if values.size != expected_size:
        raise ValueError(
            f"{filename}: found {values.size} grid values, expected "
            f"{expected_size} for grid {tuple(shape)}"
        )

    return Cube(
        natoms=natoms,
        origin=origin,
        grid_counts=tuple(grid_counts),
        shape=tuple(shape),
        axes=np.asarray(axes, dtype=float),
        atom_lines=atom_lines,
        data=values.reshape(tuple(shape), order="C"),
    )


def assert_same_grid(cubes: Sequence[Cube], atol: float = 1.0e-10) -> None:
    """Require that all cube fields can be combined point by point."""

    if not cubes:
        raise ValueError("No cube files were supplied")

    reference = cubes[0]

    for index, cube in enumerate(cubes[1:], start=2):
        if (
            cube.natoms != reference.natoms
            or cube.grid_counts != reference.grid_counts
            or cube.shape != reference.shape
        ):
            raise ValueError(f"Cube {index} has different grid dimensions")

        if not np.allclose(
            cube.origin, reference.origin, rtol=0.0, atol=atol
        ):
            raise ValueError(f"Cube {index} has a different grid origin")

        if not np.allclose(
            cube.axes, reference.axes, rtol=0.0, atol=atol
        ):
            raise ValueError(f"Cube {index} has different grid vectors")

        if cube.atom_lines != reference.atom_lines:
            raise ValueError(f"Cube {index} has a different atom block")


def _check_forte2_modulus(cube: Cube, label: str) -> None:
    """Check the non-negativity expected from np.abs(...) output."""

    minimum = float(np.min(cube.data))
    if minimum < -1.0e-12:
        raise ValueError(
            f"{label} contains negative values (minimum={minimum:.6e}); "
            "it is not a Forte2 default modulus cube"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    spinor_parser = subparsers.add_parser(
        "spinor",
        help="convert one alpha/beta spinor into one density cube",
    )
    spinor_parser.add_argument("alpha_cube")
    spinor_parser.add_argument("beta_cube")
    spinor_parser.add_argument("output_cube")
    spinor_parser.add_argument(
        "--iso-fraction",
        type=float,
        default=0.02,
        help="suggested VMD isovalue as a fraction of max density",
    )

    pair_parser = subparsers.add_parser(
        "kpair",
        help="convert two Kramers-partner spinors into one averaged cube",
    )
    pair_parser.add_argument("p_alpha_cube")
    pair_parser.add_argument("p_beta_cube")
    pair_parser.add_argument("pbar_alpha_cube")
    pair_parser.add_argument("pbar_beta_cube")
    pair_parser.add_argument("output_cube")
    pair_parser.add_argument(
        "--iso-fraction",
        type=float,
        default=0.02,
        help="suggested VMD isovalue as a fraction of max density",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not 0.0 < args.iso_fraction < 1.0:
        raise ValueError("--iso-fraction must lie strictly between 0 and 1")

    if args.command == "spinor":
        make_spinor_density_cube(
            args.alpha_cube,
            args.beta_cube,
            args.output_cube,
            iso_fraction=args.iso_fraction,
        )
        return

    if args.command == "kpair":
        make_kramers_pair_density_cube(
            args.p_alpha_cube,
            args.p_beta_cube,
            args.pbar_alpha_cube,
            args.pbar_beta_cube,
            args.output_cube,
            iso_fraction=args.iso_fraction,
        )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
