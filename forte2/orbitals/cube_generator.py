import math
from pathlib import Path
import numpy as np

from forte2.helpers import logger
from forte2.lib import ints


def read_cube(filename):
    filename = Path(filename)
    lines = filename.read_text().splitlines()

    # read the number of atoms from the third line
    natoms = abs(int(lines[2].split()[0]))

    # standard cube header:
    # 2 comment lines + 1 line for origin + 3 lines for grid + natoms lines for atoms
    data_start = 6 + natoms

    header = lines[:data_start]

    values = np.fromstring(
        " ".join(lines[data_start:]),
        sep=" ",
    )

    # read the number of grids
    nx = abs(int(lines[3].split()[0]))
    ny = abs(int(lines[4].split()[0]))
    nz = abs(int(lines[5].split()[0]))

    expected_size = nx * ny * nz

    if values.size != expected_size:
        raise ValueError(
            f"{filename.name}: found {values.size} values, "
            f"but expected {expected_size}."
        )

    return {
        "filename": filename,
        "header": header,
        "values": values,
        "shape": (nx, ny, nz),
        "geometry_header": lines[2:data_start],
    }


def write_cube(filename, header, values):
    filename = Path(filename)

    with filename.open("w") as handle:
        handle.write(
            "GHF spinor orbital total density\n"
        )
        handle.write(
            "|psi_a|^2 + |psi_b|^2\n"
        )

        for line in header[2:]:
            handle.write(line + "\n")

        # Gaussian cube files have 6 values per line
        for start in range(0, values.size, 6):
            block = values[start:start + 6]

            handle.write(
                " ".join(
                    f"{value:13.6E}"
                    for value in block
                )
                + "\n"
            )


def combine_spinor_density(
    alpha_cube,
    beta_cube,
    output_cube,
):
    cube_a = read_cube(alpha_cube)
    cube_b = read_cube(beta_cube)

    # check net grid
    if cube_a["shape"] != cube_b["shape"]:
        raise ValueError(
            "Alpha and beta cubes have different grid sizes: "
            f"{cube_a['shape']} versus {cube_b['shape']}."
        )

    # check geometry header
    if cube_a["geometry_header"] != cube_b["geometry_header"]:
        raise ValueError(
            "Alpha and beta cube headers are not compatible."
        )

    psi_a_abs = cube_a["values"]
    psi_b_abs = cube_b["values"]

    density = (
        psi_a_abs**2
        + psi_b_abs**2
    )

    write_cube(
        output_cube,
        cube_a["header"],
        density,
    )

    print("Grid:", cube_a["shape"])
    print("Density minimum:", density.min())
    print("Density maximum:", density.max())
    print("Output:", output_cube)

from pathlib import Path
import re


def combine_all_spinor_cubes(
    directory,
    output_subdir="total_density",
):
    """

    Parameters
    ----------
    directory: the directory containing the cube files to combine. 
    The function will look for pairs of files with the pattern {base}_{spin}.cube, 
    where {spin} is either 'a' or 'b'. 
    For each pair, it will generate a combined total-density cube file named {base}_total_density.cube.
    output_subdir: subdirectory created under directory to store combined cube files.
    ----------
    """
    directory = Path(directory)
    output_directory = directory / output_subdir
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # match files with the pattern: {base}_{spin}.cube
    # where {spin} is either 'a' or 'b'
    pattern = re.compile(
        r"^(?P<base>.+)_(?P<spin>[ab])\.cube$"
    )

    pairs = {}

    for cube_path in directory.glob("*.cube"):
        match = pattern.fullmatch(cube_path.name)

        if match is None:
            continue

        base = match.group("base")
        spin = match.group("spin")

        if base not in pairs:
            pairs[base] = {}

        pairs[base][spin] = cube_path

    combined_count = 0

    for base, components in sorted(pairs.items()):
        alpha_path = components.get("a")
        beta_path = components.get("b")

        # check if both components exist
        if alpha_path is None:
            print(f"Skip {base}: missing _a.cube")
            continue

        if beta_path is None:
            print(f"Skip {base}: missing _b.cube")
            continue

        output_path = (
            output_directory
            / f"{base}_total_density.cube"
        )

        print(f"Combining orbital: {base}")
        print(f"  alpha : {alpha_path.name}")
        print(f"  beta  : {beta_path.name}")
        print(f"  output: {output_path.name}")

        combine_spinor_density(
            alpha_cube=alpha_path,
            beta_cube=beta_path,
            output_cube=output_path,
        )

        combined_count += 1

    print(
        f"\nFinished: generated "
        f"{combined_count} total-density cube files."
    )



def write_orbital_cubes(
    system,
    C,
    indices=None,
    prefix="orbital",
    filepath=".",
    format=("cube",),
    spacing=0.2,
    padding=4.0,
):
    """A convenience function to write cube files for molecular orbitals.

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
    format : Tuple[str, ...] or str, optional, default=("cube",)
        Output formats to generate. Supported values:
          - "cube": standard CUBE files. For two-component systems, writes
            separate alpha and beta magnitude files ("_a.cube" and "_b.cube").
          - "2ccube": two-component cube files (".2ccube"). For two-component
            systems only, writes four datasets consecutively per orbital in the
            order: alpha real, alpha imag, beta real, beta imag.
        Multiple formats can be requested at once, e.g., formats=("cube","2ccube").
    spacing : float, optional, default=0.2
        The spacing between grid points in the cube file (in bohr).
    padding : float, optional, default=4.0
        The extra space (in bohr) added in all directions around the atoms when generating the grid.

    Usage
    -----
    >>> write_orbital_cubes(system, C, indices=[0,1,2], prefix="orbital", filepath="cubes/", format=("cube","2ccube"))
    """
    # if format is a single string, convert it to a tuple
    if isinstance(format, str):
        format = (format,)
    gen = CubeGenerator(spacing=spacing, padding=padding)
    return gen.write_cubes(
        system, C, indices=indices, prefix=prefix, filepath=filepath, formats=format
    )


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

    def write_cubes(
        self,
        system,
        C,
        indices=None,
        prefix="orbital",
        filepath=".",
        formats=("cube",),
    ) -> None:
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
        formats : Tuple[str, ...], optional, default=("cube",)
            Output formats to generate. Supported values:
              - "cube": standard CUBE files. For two-component systems, writes
                separate alpha and beta magnitude files ("_a.cube" and "_b.cube").
              - "2ccube": two-component cube files (".2ccube"). For two-component
                systems only, writes four datasets consecutively per orbital in the
                order: alpha real, alpha imag, beta real, beta imag.
            Multiple formats can be requested at once, e.g., formats=("cube","2ccube").
        """

        filepath = Path(filepath)

        # validate formats
        supported_formats = {"cube", "2ccube"}
        invalid_formats = set(formats) - supported_formats
        if invalid_formats:
            logger.log(f"Warning: Ignoring unsupported cube formats: {invalid_formats}")

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
            # two-component systems
            # 1) standard cube magnitude outputs if requested
            if "cube" in formats:
                for i, index in enumerate(indices):
                    cube_a = self._make_cube(
                        np.abs(values_a[:, i]),
                        grid_origin,
                        npoints,
                        scaled_axes,
                        system,
                    )
                    _write_file(
                        cube_a,
                        filepath,
                        f"{prefix}_{index:0{number_of_digits}d}_a.cube",
                    )

                    cube_b = self._make_cube(
                        np.abs(values_b[:, i]),
                        grid_origin,
                        npoints,
                        scaled_axes,
                        system,
                    )
                    _write_file(
                        cube_b,
                        filepath,
                        f"{prefix}_{index:0{number_of_digits}d}_b.cube",
                    )

            # 2) two-component four-field output if requested
            if "2ccube" in formats:
                # ensure we have separate real/imag pieces
                # If the coefficient matrix is real, imag parts are zeros.
                have_complex = np.iscomplexobj(values_a) or np.iscomplexobj(values_b)
                for i, index in enumerate(indices):
                    a_re = values_a[:, i].real.astype(float)
                    a_im = (
                        values_a[:, i].imag.astype(float)
                        if have_complex
                        else np.zeros_like(a_re)
                    )
                    b_re = values_b[:, i].real.astype(float)
                    b_im = (
                        values_b[:, i].imag.astype(float)
                        if have_complex
                        else np.zeros_like(b_re)
                    )

                    twocc = self._make_2ccube(
                        a_re,
                        a_im,
                        b_re,
                        b_im,
                        grid_origin,
                        npoints,
                        scaled_axes,
                        system,
                    )
                    _write_file(
                        twocc,
                        filepath,
                        f"{prefix}_{index:0{number_of_digits}d}.2ccube",
                    )

        else:
            # one-component systems
            if "cube" in formats:
                for i, index in enumerate(indices):
                    cube = self._make_cube(
                        values[:, i], grid_origin, npoints, scaled_axes, system
                    )
                    _write_file(
                        cube, filepath, f"{prefix}_{index:0{number_of_digits}d}.cube"
                    )
            if "2ccube" in formats:
                logger.log(
                    "Requested '2ccube' output for a one-component system; skipping this format."
                )

    def _make_header_and_coords(self, minr, npoints, axis, atoms, label):
        """
        Create the header and atom section of the cube file.
        """
        header = f"""{label}

{len(atoms):6d} {minr[0]:10.6f} {minr[1]:10.6f} {minr[2]:10.6f}
{npoints[0]:6d} {axis[0][0]:10.6f} {axis[0][1]:10.6f} {axis[0][2]:10.6f}
{npoints[1]:6d} {axis[1][0]:10.6f} {axis[1][1]:10.6f} {axis[1][2]:10.6f}
{npoints[2]:6d} {axis[2][0]:10.6f} {axis[2][1]:10.6f} {axis[2][2]:10.6f}"""

        coords = "\n".join(
            f"{Z:3d} {0.0:10.6f} {x:10.6f} {y:10.6f} {z:10.6f}"
            for Z, (x, y, z) in atoms
        )
        return header, coords

    def _make_cube(self, values, minr, npoints, axis, system):
        """
        Create a cube file from the values on a grid.
        """
        # write the cube file
        header, coords = self._make_header_and_coords(
            minr, npoints, axis, system.atoms, "Forte2 Cube File"
        )

        v = values.flatten()
        lines = [
            " ".join(f"{x:.5E}" for x in v[i : i + 6]) for i in range(0, len(v), 6)
        ]
        return header + "\n" + coords + "\n" + "\n".join(lines)

    def _make_2ccube(self, a_re, a_im, b_re, b_im, minr, npoints, axis, system):
        """
        Create a two-component cube file containing four datasets written consecutively
        in the order: alpha real, alpha imag, beta real, beta imag.
        """
        header, coords = self._make_header_and_coords(
            minr, npoints, axis, system.atoms, "Forte2 2-Component Cube File"
        )

        v = np.concatenate(
            [
                a_re.flatten(),
                a_im.flatten(),
                b_re.flatten(),
                b_im.flatten(),
            ]
        )
        lines = [
            " ".join(f"{x:.5E}" for x in v[i : i + 6]) for i in range(0, len(v), 6)
        ]
        return header + "\n" + coords + "\n" + "\n".join(lines)
