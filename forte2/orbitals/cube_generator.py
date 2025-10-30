import math
from pathlib import Path
import numpy as np

from forte2.helpers import logger
from forte2 import ints


def simple_grid(
    atoms, spacing: tuple[float, float, float], padding: tuple[float, float, float]
):
    """
    Create a simple cubic grid around the given atoms.

    Parameters
    ----------
    atoms : List[Tuple[int, Tuple[float, float, float]]]
        List of atoms, each represented as a tuple (Z, (x, y, z)).
    spacing : Tuple[float, float, float]
        The spacing between grid points in the x, y, and z directions.
    padding : Tuple[float, float, float]
        The amount of padding (extra space) to add to the grid in the x, y, and z directions around the atoms when generating the grid.

    Returns
    -------
    grid_origin : Tuple[float, float, float]
        The origin of the grid.
    npoints : Tuple[int, int, int]
        The number of grid points in the x, y, and z directions.
    scaled_axes : List[Tuple[float, float, float]]
        The scaled axes for the grid.
    """

    # find the extents of the molecule based on atom positions
    xrange = (math.inf, -math.inf)
    yrange = (math.inf, -math.inf)
    zrange = (math.inf, -math.inf)
    for _, (x, y, z) in atoms:
        xrange = (min(xrange[0], x), max(xrange[1], x))
        yrange = (min(yrange[0], y), max(yrange[1], y))
        zrange = (min(zrange[0], z), max(zrange[1], z))

    # add padding
    xrange = (xrange[0] - padding[0], xrange[1] + padding[0])
    yrange = (yrange[0] - padding[1], yrange[1] + padding[1])
    zrange = (zrange[0] - padding[2], zrange[1] + padding[2])

    # compute the number of points
    npoints = (
        math.ceil((r[1] - r[0]) / s)
        for (r, s) in zip((xrange, yrange, zrange), spacing)
    )

    return (
        (xrange[0], yrange[0], zrange[0]),
        tuple(npoints),
        [(spacing[0], 0, 0), (0, spacing[1], 0), (0, 0, spacing[2])],
    )


class CubeGenerator:
    """
    Class to generate cube files from a given set of molecular orbitals.

    Parameters
    ----------
    spacing : float, optional, default=0.2
        The spacing between grid points in the cube file (in bohr).
    padding : float, optional, default=4.0
        The extra space (in bohr) added in all directions around the atoms when generating the grid.

    Usage
    -----
    ```
    cube = CubeGenerator()
    cube.run(system, C)
    ```

    """

    def __init__(self, spacing=0.2, padding=4.0):
        self.spacing = [spacing, spacing, spacing]
        self.padding = [padding, padding, padding]

    def run(self, system, C, indices=None, prefix="orbital", filepath=".") -> None:
        """
        Generate cube files for the given orbitals.

        This method generates cube files for the specified molecular orbitals
        represented by the coefficient matrix `C`. The cube files are saved in the
        specified directory with the given prefix. The files are named as:

            {prefix}_{index:0{number_of_digits}d}.cube

        where `index` is the zero-based index of the orbital and `number_of_digits` is
        determined by the maximum index to ensure proper zero-padding.

        Parameters
        ----------
        system : forte2.system.System
            The system object containing the molecular information.
        C : NDArray
            The molecular orbital coefficients matrix.
        indices : List[int], optional, default=None
            The indices (zero-based) of the orbitals to generate cube files for.
            By default all orbitals are generated.
        prefix : str, optional, default="orbital"
            The prefix for the cube file names.
        filepath : str, optional, default="."
            The directory to save the cube files in.
        """

        filepath = Path(filepath)
        # determine the indices of the orbitals to generate
        indices = indices if indices is not None else range(C.shape[1])
        max_index = max(indices)
        number_of_digits = int(math.log10(max_index + 1)) + 1

        # determine the grid points for the cube file
        grid_origin, npoints, scaled_axes = simple_grid(
            system.atoms, spacing=self.spacing, padding=self.padding
        )

        logger.log(f"\nGenerating cube files with the following parameters:")
        logger.log(
            f"  Grid origin: ({grid_origin[0]:.3f}, {grid_origin[1]:.3f}, {grid_origin[2]:.3f})"
        )
        logger.log(f"  Grid points: {npoints[0]} x {npoints[1]} x {npoints[2]} points.")
        logger.log(f"  Scaled axes: {scaled_axes}")
        logger.log(f"  Orbitals: {list(indices)}\n")

        # calculate the orbitals on the grid
        if system.two_component:
            # for two-component systems, we need to calculate the orbitals for both alpha and beta
            nbf = system.nbf
            values_a = ints.orbitals_on_grid(
                system.basis,
                C[:nbf, indices].real,
                grid_origin,
                npoints,
                scaled_axes,
            )
            values_b = ints.orbitals_on_grid(
                system.basis,
                C[nbf:, indices].real,
                grid_origin,
                npoints,
                scaled_axes,
            )
            if C.dtype == complex:
                values_a = values_a.astype(complex)
                values_b = values_b.astype(complex)
                values_a += 1.0j * ints.orbitals_on_grid(
                    system.basis,
                    C[:nbf, indices].imag,
                    grid_origin,
                    npoints,
                    scaled_axes,
                )
                values_b += 1.0j * ints.orbitals_on_grid(
                    system.basis,
                    C[nbf:, indices].imag,
                    grid_origin,
                    npoints,
                    scaled_axes,
                )
        else:
            values = ints.orbitals_on_grid(
                system.basis,
                C[:, indices],
                grid_origin,
                npoints,
                scaled_axes,
            )

        def _write_file(cube, filepath, filename):
            # check if the directory exists, if not, create it
            filepath = Path(filepath)
            filepath.mkdir(parents=True, exist_ok=True)
            with open(filepath / filename, "w") as f:
                f.write(cube)

        # write the cube files
        if system.two_component:
            # for two-component systems, write the magnitude of alpha and beta parts in separate cube files
            for i, index in enumerate(indices):
                cube_a = self._make_cube(
                    np.abs(values_a[:, i]), grid_origin, npoints, scaled_axes, system
                )
                _write_file(
                    cube_a,
                    filepath,
                    f"{prefix}_{index:0{number_of_digits}d}_a.cube",
                )

                cube_b = self._make_cube(
                    np.abs(values_b[:, i]), grid_origin, npoints, scaled_axes, system
                )
                _write_file(
                    cube_b,
                    filepath,
                    f"{prefix}_{index:0{number_of_digits}d}_b.cube",
                )

        else:
            for i, index in enumerate(indices):
                cube = self._make_cube(
                    values[:, i], grid_origin, npoints, scaled_axes, system
                )
                _write_file(
                    cube, filepath, f"{prefix}_{index:0{number_of_digits}d}.cube"
                )

    def _make_cube(self, values, minr, npoints, axis, system):
        """
        Create a cube file from the values.
        """
        # write the cube file
        header = f"""Forte2 Cube File.

{len(system.atoms):6d} {minr[0]:10.6f} {minr[1]:10.6f} {minr[2]:10.6f}
{npoints[0]:6d} {axis[0][0]:10.6f} {axis[0][1]:10.6f} {axis[0][2]:10.6f}
{npoints[1]:6d} {axis[1][0]:10.6f} {axis[1][1]:10.6f} {axis[1][2]:10.6f}
{npoints[2]:6d} {axis[2][0]:10.6f} {axis[2][1]:10.6f} {axis[2][2]:10.6f}"""

        atoms = "\n".join(
            f"{Z:3d} {0.0:10.6f} {x:10.6f} {y:10.6f} {z:10.6f}"
            for Z, (x, y, z) in system.atoms
        )

        v = values.flatten()
        lines = [
            " ".join(f"{x:.5E}" for x in v[i : i + 6]) for i in range(0, len(v), 6)
        ]
        return header + "\n" + atoms + "\n" + "\n".join(lines)
