import glob
from pathlib import Path as FilePath
from matplotlib.path import Path as MplPath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from matplotlib import offsetbox
from matplotlib.patches import PathPatch
from PIL import Image
import logging
from forte2.utils.density_cube import Cube, read_cube, assert_same_grid ,_check_forte2_modulus

from forte2.orbitals import write_orbital_cubes



def write_cube(
    filename: str | FilePath,
    template: Cube,
    data: np.ndarray,
    title: str,
) -> None:
    """Write a real scalar field with the template geometry and grid."""

    filename = FilePath(filename)
    data = np.asarray(data, dtype=float)

    if data.shape != template.shape:
        raise ValueError(
            f"Output shape {data.shape} does not match {template.shape}"
        )

    with filename.open("w") as handle:
        handle.write(title.rstrip() + "\n")
        handle.write("Generated from Forte2 |alpha| and |beta| cubes\n")

        handle.write(
            f"{template.natoms:5d}"
            + "".join(f"{x:13.6f}" for x in template.origin)
            + "\n"
        )

        for count, axis in zip(template.grid_counts, template.axes):
            handle.write(
                f"{count:5d}"
                + "".join(f"{x:13.6f}" for x in axis)
                + "\n"
            )

        for line in template.atom_lines:
            handle.write(line.rstrip() + "\n")

        flat = data.ravel(order="C")
        for start in range(0, flat.size, 6):
            handle.write(
                "".join(f"{x:16.8E}" for x in flat[start : start + 6])
                + "\n"
            )


def spinor_density(
    alpha_modulus_cube: Cube,
    beta_modulus_cube: Cube,
) -> np.ndarray:
    """Return rho = |phi_alpha|^2 + |phi_beta|^2."""

    assert_same_grid([alpha_modulus_cube, beta_modulus_cube])
    _check_forte2_modulus(alpha_modulus_cube, "alpha cube")
    _check_forte2_modulus(beta_modulus_cube, "beta cube")

    return alpha_modulus_cube.data**2 + beta_modulus_cube.data**2


def print_density_diagnostics(
    density: np.ndarray,
    template: Cube,
    iso_fraction: float,
) -> None:
    """Print normalization and a convenient initial VMD isovalue."""

    integral = float(np.sum(density) * template.voxel_volume)
    maximum = float(np.max(density))

    print(f"density integral       = {integral:.10e}")
    print(f"density minimum        = {np.min(density):.10e}")
    print(f"density maximum        = {maximum:.10e}")
    print(f"suggested VMD isovalue = {iso_fraction * maximum:.10e}")


def make_spinor_density_cube(
    alpha_filename: str | FilePath,
    beta_filename: str | FilePath,
    output_filename: str | FilePath,
    iso_fraction: float = 0.02,
) -> np.ndarray:
    """Generate one scalar-density cube for one Forte2 spinor."""

    alpha_cube = read_cube(alpha_filename)
    beta_cube = read_cube(beta_filename)

    density = spinor_density(alpha_cube, beta_cube)

    write_cube(
        output_filename,
        template=alpha_cube,
        data=density,
        title="Forte2 spinor density: |alpha|^2 + |beta|^2",
    )

    print(f"wrote {output_filename}")
    print_density_diagnostics(density, alpha_cube, iso_fraction)

    return density


def make_kramers_pair_density_cube(
    p_alpha_filename: str | FilePath,
    p_beta_filename: str | FilePath,
    pbar_alpha_filename: str | FilePath,
    pbar_beta_filename: str | FilePath,
    output_filename: str | FilePath,
    iso_fraction: float = 0.02,
) -> np.ndarray:
    """Generate one averaged scalar-density cube for one Kramers pair."""

    p_alpha = read_cube(p_alpha_filename)
    p_beta = read_cube(p_beta_filename)
    pbar_alpha = read_cube(pbar_alpha_filename)
    pbar_beta = read_cube(pbar_beta_filename)

    assert_same_grid([p_alpha, p_beta, pbar_alpha, pbar_beta])

    rho_p = spinor_density(p_alpha, p_beta)
    rho_pbar = spinor_density(pbar_alpha, pbar_beta)
    rho_pair = 0.5 * (rho_p + rho_pbar)

    write_cube(
        output_filename,
        template=p_alpha,
        data=rho_pair,
        title="Forte2 Kramers-pair averaged scalar density",
    )

    dv = p_alpha.voxel_volume
    average_norm = 0.5 * np.sum(rho_p + rho_pbar) * dv
    if average_norm > 0.0:
        relative_l1_difference = (
            np.sum(np.abs(rho_p - rho_pbar)) * dv / average_norm
        )
    else:
        relative_l1_difference = float("nan")

    print(f"wrote {output_filename}")
    print_density_diagnostics(rho_pair, p_alpha, iso_fraction)
    print(
        "relative pair-density L1 difference = "
        f"{relative_l1_difference:.10e}"
    )

    return rho_pair



def find_forte2_spinor_component_cubes(
    cubedir,
    prefix="orbital",
):
    """
    Find Forte2 component cubes.

    Expected Forte2 filenames:
        orbital_0_a.cube
        orbital_0_b.cube
        orbital_01_a.cube
        orbital_01_b.cube

    Returns
    -------
    component_files : dict
        Example:
        {
            0: {
                "a": FilePath("orbital_0_a.cube"),
                "b": FilePath("orbital_0_b.cube"),
            },
            ...
        }
    """
    cubedir = FilePath(cubedir)
    import re
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(\d+)_([ab])\.cube$",
        flags=re.IGNORECASE,
    )

    component_files = {}

    for cube_file in cubedir.glob("*.cube"):
        match = pattern.fullmatch(cube_file.name)

        if match is None:
            continue

        orbital_index = int(match.group(1))
        component = match.group(2).lower()

        component_files.setdefault(orbital_index, {})

        if component in component_files[orbital_index]:
            previous = component_files[orbital_index][component]

            raise RuntimeError(
                "Duplicate Forte2 component cubes detected for "
                f"orbital {orbital_index}, component {component}:\n"
                f"    {previous}\n"
                f"    {cube_file}"
            )

        component_files[orbital_index][component] = cube_file

    return component_files

def generate_kramers_pair_density_cubes(
    component_cubedir,
    kramers_pairs,
    output_cubedir,
    prefix="orbital",
    iso_fraction=0.02,
):
    """
    Generate one averaged density cube for each Kramers pair.

    For P = (p, pbar):

        rho_P(r) = 0.5 * [rho_p(r) + rho_pbar(r)]

    where:

        rho_p(r) = |alpha_p(r)|^2 + |beta_p(r)|^2

    Returns
    -------
    pair_density_cubes : dict

        {
            (p, pbar): FilePath("kpair_p_pbar_total_density.cube"),
            ...
        }
    """
    component_cubedir = FilePath(component_cubedir)
    output_cubedir = FilePath(output_cubedir)

    output_cubedir.mkdir(parents=True, exist_ok=True)

    component_files = find_forte2_spinor_component_cubes(
        component_cubedir,
        prefix=prefix,
    )

    pair_density_cubes = {}

    for pair_number, pair in enumerate(kramers_pairs):
        if len(pair) != 2:
            raise ValueError(
                f"kramers_pairs[{pair_number}] must contain exactly "
                "two spinor indices."
            )

        p = int(pair[0])
        pbar = int(pair[1])

        if p == pbar:
            raise ValueError(
                f"Kramers pair {pair_number} contains the same "
                f"spinor twice: ({p}, {pbar})."
            )

        # 
        missing_files = []

        for orbital_index in (p, pbar):
            for component in ("a", "b"):
                if component not in component_files.get(
                    orbital_index, {}
                ):
                    missing_files.append(
                        f"{prefix}_{orbital_index}_{component}.cube"
                    )

        if missing_files:
            raise FileNotFoundError(
                "Cannot construct Kramers-pair density. "
                "Missing component cube(s): "
                + ", ".join(missing_files)
            )

        p_alpha = component_files[p]["a"]
        p_beta = component_files[p]["b"]

        pbar_alpha = component_files[pbar]["a"]
        pbar_beta = component_files[pbar]["b"]

        output_cube = (
            output_cubedir
            / f"kpair_{p}_{pbar}_total_density.cube"
        )

        make_kramers_pair_density_cube(
            p_alpha,
            p_beta,
            pbar_alpha,
            pbar_beta,
            output_cube,
            iso_fraction=iso_fraction,
        )

        pair_density_cubes[(p, pbar)] = output_cube

    return pair_density_cubes



def find_vmd_rendered_image(cube_file):
    """
    Find the VMDCube image corresponding exactly to cube_file.

    For:
        kpair_0_1_total_density.cube

    searches:
        kpair_0_1_total_density.tga
        kpair_0_1_total_density.png
    """
    cube_file = FilePath(cube_file)

    candidates = [
        cube_file.with_suffix(".tga"),
        cube_file.with_suffix(".png"),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "VMDCube did not generate an image corresponding to:\n"
        f"    {cube_file}\n"
        "Expected one of:\n"
        + "\n".join(f"    {path}" for path in candidates)
    )




def get_color_and_alpha_smooth(value, vmin, vmax, cmap):
    """
    Map a value in [vmin, vmax] to a color using a continuous colormap
    and an alpha using logarithmic scaling.

    Parameters
    ----------
    value : float
        The value to be mapped.
    vmin : float
        Minimum value for normalization.
    vmax : float
        Maximum value for normalization.
    cmap : matplotlib.colors.Colormap
        The colormap to use.

    Returns
    -------
    color : tuple
        RGBA tuple from the colormap.
    alpha : float
        Transparency in [0, 1], scaled logarithmically.
    """
    # clamp value to [vmin, vmax]
    value = float(np.clip(value, vmin, vmax))

    # Logarithmic normalization
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    color = cmap(norm(value))  # RGBA
    alpha = float(norm(value))  # in [0, 1]

    return color, alpha


def plot_smooth_connection(
    ax, x_coords, y_coords, i, j, val, vmin, vmax, cmap="magma_r"
):
    """
    Plots a smooth Bezier curve between two points (i and j) on the mutual
    correlation plot with a given color and transparency.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x_coords : list of float
        x-coordinates of the points.
    y_coords : list of float
        y-coordinates of the points.
    i : int
        Index of the first point.
    j : int
        Index of the second point.
    val : float
        Value used to determine color and transparency.
    vmin : float
        Minimum value for normalization.
    vmax : float
        Maximum value for normalization.
    cmap : str
        Name of the matplotlib colormap to use.
    """

    color, alpha = get_color_and_alpha_smooth(val, vmin, vmax, cmap)

    # Define the three points
    p0 = [x_coords[i], y_coords[i]]
    p1 = [
        0.1 * (x_coords[i] + x_coords[j]),
        0.1 * (y_coords[i] + y_coords[j]),
    ]
    p2 = [x_coords[j], y_coords[j]]

    # Create a Matplotlib path for a quadratic Bezier curve
    verts = [p0, p1, p2]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]

    path = MplPath(verts, codes)
    patch = PathPatch(
        path, facecolor="none", edgecolor=color, lw=1 + 3 * alpha, alpha=alpha
    )
    ax.add_patch(patch)


def mutual_correlation_plot_from_values(
    system,
    C,
    indices,
    mutual_correlation_matrix,
    orbital_occupations,
    rotation_matrix=None,
    orbitals_filepath="mca_orbitals",
    radius=1.0,
    offset=1.5,
    zoom=0.2,
    fontsize=10,
    figsize=(6, 6),
    output_file=None,
    vmin=0.00075,
    vmax=0.75,
    cmap_name="magma_r",
    show_colorbar=True,
    vmd_parameters=None,
):
    """
    Plot a mutual-correlation network from explicitly supplied values.

    This function is independent of ``MutualCorrelationAnalysis`` and is intended
    for cases where you already computed the mutual correlation matrix ``M2``
    and the corresponding orbital occupations in a custom basis (for example,
    a natural-orbital basis).

    Parameters
    ----------
    system : System
        The Forte2 System object.
    C : NDArray
        The molecular orbital coefficients.
    indices : List[int]
        List of orbital indices to plot.
    mutual_correlation_matrix : NDArray
        The mutual correlation matrix to visualize. Expected shape is
        ``(norb, norb)``.
    orbital_occupations : NDArray
        Orbital occupations used for the orbital labels. Expected shape is
        ``(norb,)``.
    rotation_matrix : NDArray, optional
        Orthogonal rotation matrix used to transform the mutual-correlation
        matrix from the MO basis to the NO basis. If provided, the matrix is
        applied as ``M2_no = Q.T @ M2_mo @ Q``.
    orbitals_filepath : str, optional, default="mca_orbitals"
        Directory to save orbital cube files.
    radius : float, optional, default=1.0
        Radius of the circle on which orbitals are placed.
    offset : float, optional, default=1.5
        Offset for placing orbital images.
    zoom : float, optional, default=0.2
        Zoom factor for orbital images.
    fontsize : int, optional, default=10
        Font size for labels.
    figsize : Tuple[float, float], optional, default=(6, 6)
        Size of the figure.
    output_file : str, optional
        If provided, saves the plot to a file with this name (png format).
    vmin : float, optional, default=0.00075
        Minimum value for color mapping.
    vmax : float, optional, default=0.75
        Maximum value for color mapping.
    cmap_name : str, optional, default="magma_r"
        Name of the matplotlib colormap to use.
    show_colorbar : bool, optional, default=True
        Whether to display the colorbar.
    vmd_parameters : dict, optional
        Parameters to pass to VMDCube for orbital visualization.
    """

    mutual_correlation_matrix = np.asarray(mutual_correlation_matrix, dtype=float)
    orbital_occupations = np.asarray(orbital_occupations, dtype=float)

    if mutual_correlation_matrix.ndim != 2:
        raise ValueError("mutual_correlation_matrix must be a 2D array.")

    num_orbitals = len(indices)
    if mutual_correlation_matrix.shape != (num_orbitals, num_orbitals):
        raise ValueError(
            f"mutual_correlation_matrix must have shape {(num_orbitals, num_orbitals)}, got {mutual_correlation_matrix.shape}."
        )

    if orbital_occupations.shape != (num_orbitals,):
        raise ValueError(
            f"orbital_occupations must have shape ({num_orbitals},), got {orbital_occupations.shape}."
        )

    if rotation_matrix is not None:
        rotation_matrix = np.asarray(rotation_matrix, dtype=float)
        if rotation_matrix.shape != (num_orbitals, num_orbitals):
            raise ValueError(
                f"rotation_matrix must have shape {(num_orbitals, num_orbitals)}, got {rotation_matrix.shape}."
            )
        mutual_correlation_matrix = rotation_matrix.T @ mutual_correlation_matrix @ rotation_matrix

    # generate cube files for the orbitals
    write_orbital_cubes(
        system, C, indices=indices, filepath=orbitals_filepath, prefix="orbital"
    )
    from forte2.orbitals.cube_generator import combine_all_spinor_cubes
    combine_all_spinor_cubes(orbitals_filepath, output_subdir="combined_cubes")
    orbitals_filepath = f"{orbitals_filepath}/combined_cubes"
    # run VMDCube
    try:
        from vmdcube import VMDCube
    except ImportError:
        raise ImportError("VMDCube is not installed")

    vmd_parameters = {} if vmd_parameters is None else vmd_parameters

    vmd = VMDCube(cubedir=orbitals_filepath, **vmd_parameters)
    vmd.run()

    # Set font types for better compatibility
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    # get the color map
    cmap = colormaps[cmap_name]

    # 1) Place orbitals on a circle
    angles = np.linspace(0, 2 * np.pi, num_orbitals, endpoint=False)
    x_coords = radius * np.sin(angles)
    y_coords = radius * np.cos(angles)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")

    # find all the files with the pattern h2_ghf_orbs_*_total_density.tga
    orbitals_filepath = FilePath(orbitals_filepath)

    # form a dictionary mapping orbital index (int) to tga file path
    tga_files = glob.glob(
        str(orbitals_filepath / FilePath("orbital_*_total_density.tga"))
    )
    tga_files_dict = {}
    for file in tga_files:
        name = FilePath(file).stem
        if not cube_nonzero(f"{orbitals_filepath}/{name}.cube"):
            print(name, "is zero, skipping image.")
            continue
        import re
        match = re.fullmatch(r"orbital_(\d+)_total_density", name)

        if match:
            orbital_index = int(match.group(1))
            tga_files_dict[orbital_index] = file
    # print("indice_dict",tga_files_dict.keys()
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        # print("indices", indices[i])
        orbital_index = indices[i]
        
        x_img = (radius + offset) * x / radius
        y_img = (radius + offset) * y / radius

        filename = tga_files_dict[orbital_index]

        tga_file = filename
        try:
            # Load the image
            img = Image.open(tga_file)

            # Convert that to an OffsetImage object and set the zoom
            imagebox = offsetbox.OffsetImage(img, zoom=zoom)

            # Create an AnnotationBbox to place the image at (x_img, y_img)
            ab = offsetbox.AnnotationBbox(
                imagebox,
                (x_img, y_img),
                frameon=False,
            )
            ax.add_artist(ab)

        except FileNotFoundError:
            # If the file doesn't exist, just skip
            print(f"Warning: Could not find file {tga_file}")

    # Label each orbital with the occupation number and index
    for i in range(num_orbitals):
        val = orbital_occupations[i]
        get_color_and_alpha_smooth(val, 0.01, 2, cmap)
        ax.text(
            x_coords[i] * 1.5,
            y_coords[i] * 1.5,
            f"{val:.2f} ({indices[i]})",
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        r = 0.05
        circle = plt.Circle(
            (x_coords[i], y_coords[i]),
            r,
            alpha=1.0,
            zorder=2,
        )
        ax.add_artist(circle)

    # Plot mutual correlation connections
    for i in range(num_orbitals):
        for j in range(i + 1, num_orbitals):
            val = mutual_correlation_matrix[i, j]
            plot_smooth_connection(ax, x_coords, y_coords, i, j, val, vmin, vmax, cmap)

    # Formatting
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis("off")

    import matplotlib.colors as mcolors

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if show_colorbar:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, orientation="vertical")
        cbar.set_label("Mutual Correlation", rotation=270, labelpad=15)

    # Save the plot if a filename is provided
    if output_file:
        # suppress font warnings
        import logging

        logging.getLogger("fontTools").setLevel(logging.WARNING)
        plt.savefig(f"{output_file}.png", bbox_inches="tight")
    plt.show()

def paired_mutual_correlation_plot_from_values(
    system,
    C,
    kramers_pairs,
    mutual_correlation_matrix,
    spinor_occupations,
    orbitals_filepath="mca_orbitals",
    radius=1.0,
    offset=1.5,
    zoom=0.2,
    fontsize=10,
    figsize=(6, 6),
    output_file=None,
    vmin=0.00075,
    vmax=0.75,
    cmap_name="magma_r",
    show_colorbar=True,
    vmd_parameters=None,
    density_iso_fraction=0.02,
):
    """
    Plot a mutual-correlation network using Kramers-pair densities.

    Parameters
    ----------
    system
        Forte2 System object.

    C
        Spinor MO coefficient matrix.

    kramers_pairs
        Explicit list of Kramers pairs:

            [(p0, pbar0), (p1, pbar1), ...]

        Each pair corresponds to one node in the plot.

    mutual_correlation_matrix
        Pair-level mutual-correlation matrix.

        Shape:
            (number_of_pairs, number_of_pairs)

    spinor_occupations
        Occupation number for each spinor.

        will transform in the functions

        Shape:
            (2*number_of_pairs,)


    orbitals_filepath
        Parent directory for component and pair-density cubes.

    density_iso_fraction
        Suggested density isovalue as a fraction of maximum density.
    """

    # ------------------------------------------------------------
    # 1. Validate Kramers pairs
    # ------------------------------------------------------------

    pairs = []

    for pair_number, pair in enumerate(kramers_pairs):
        if len(pair) != 2:
            raise ValueError(
                f"kramers_pairs[{pair_number}] must contain "
                "exactly two indices."
            )

        p = int(pair[0])
        pbar = int(pair[1])

        if p == pbar:
            raise ValueError(
                f"Invalid Kramers pair: ({p}, {pbar})."
            )

        pairs.append((p, pbar))

    num_pairs = len(pairs)

    if num_pairs == 0:
        raise ValueError("kramers_pairs cannot be empty.")

    flattened_indices = [
        orbital_index
        for pair in pairs
        for orbital_index in pair
    ]

    if len(flattened_indices) != len(set(flattened_indices)):
        raise ValueError(
            "A spinor index occurs in more than one Kramers pair."
        )

    # ------------------------------------------------------------
    # 2. Validate pair-level numerical data
    # ------------------------------------------------------------

    mutual_correlation_matrix = np.asarray(
        mutual_correlation_matrix,
        dtype=float,
    )


    orbital_occupations = np.array([
        spinor_occupations[p] + spinor_occupations[pbar]
    for p, pbar in kramers_pairs
    ])

    orbital_occupations = np.asarray(
        orbital_occupations,
        dtype=float,
    )
    
    expected_matrix_shape = (num_pairs, num_pairs)

    if mutual_correlation_matrix.shape != expected_matrix_shape:
        raise ValueError(
            "mutual_correlation_matrix must be a Kramers-pair "
            f"matrix with shape {expected_matrix_shape}, got "
            f"{mutual_correlation_matrix.shape}."
        )

    if orbital_occupations.shape != (num_pairs,):
        raise ValueError(
            "orbital_occupations must contain one value per "
            f"Kramers pair. Expected shape ({num_pairs},), got "
            f"{orbital_occupations.shape}."
        )

    if not np.all(np.isfinite(mutual_correlation_matrix)):
        raise ValueError(
            "mutual_correlation_matrix contains NaN or infinity."
        )

    if not np.all(np.isfinite(orbital_occupations)):
        raise ValueError(
            "orbital_occupations contains NaN or infinity."
        )

    if not np.allclose(
        mutual_correlation_matrix,
        mutual_correlation_matrix.T,
        rtol=1.0e-8,
        atol=1.0e-12,
    ):
        raise ValueError(
            "mutual_correlation_matrix must be symmetric."
        )

    if vmin <= 0.0 or vmax <= vmin:
        raise ValueError(
            "Logarithmic color mapping requires 0 < vmin < vmax."
        )

    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    # ------------------------------------------------------------
    # 3. Generate Forte2 alpha/beta component cubes
    # ------------------------------------------------------------

    orbitals_filepath = FilePath(orbitals_filepath)

    component_cubedir = (
        orbitals_filepath / "spinor_components"
    )

    pair_density_cubedir = (
        orbitals_filepath / "kramers_pair_density"
    )

    component_cubedir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_density_cubedir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_orbital_cubes(
        system,
        C,
        indices=sorted(flattened_indices),
        filepath=str(component_cubedir),
        prefix="orbital",
    )

    # ------------------------------------------------------------
    # 4. Generate one density cube per Kramers pair
    # ------------------------------------------------------------

    pair_density_cubes = (
        generate_kramers_pair_density_cubes(
            component_cubedir=component_cubedir,
            kramers_pairs=pairs,
            output_cubedir=pair_density_cubedir,
            prefix="orbital",
            iso_fraction=density_iso_fraction,
        )
    )

    # ------------------------------------------------------------
    # 5. Run VMDCube only on the pair-density directory
    # ------------------------------------------------------------

    try:
        from vmdcube import VMDCube
    except ImportError as exc:
        raise ImportError(
            "VMDCube is not installed."
        ) from exc

    if vmd_parameters is None:
        vmd_parameters = {}
    else:
        vmd_parameters = dict(vmd_parameters)

    vmd = VMDCube(
        cubedir=str(pair_density_cubedir),
        **vmd_parameters,
    )

    vmd.run()

    # ------------------------------------------------------------
    # 6. Match every pair directly to its rendered image
    # ------------------------------------------------------------

    pair_images = {}

    for pair in pairs:
        density_cube = pair_density_cubes[pair]
        rendered_image = find_vmd_rendered_image(
            density_cube
        )

        pair_images[pair] = rendered_image

    # ------------------------------------------------------------
    # 7. Matplotlib configuration
    # ------------------------------------------------------------

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    cmap = colormaps[cmap_name]

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num_pairs,
        endpoint=False,
    )

    x_coords = radius * np.sin(angles)
    y_coords = radius * np.cos(angles)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_aspect("equal", "box")

    # ------------------------------------------------------------
    # 8. Place Kramers-pair density images
    # ------------------------------------------------------------

    for i, pair in enumerate(pairs):
        x = x_coords[i]
        y = y_coords[i]

        x_img = (radius + offset) * x / radius
        y_img = (radius + offset) * y / radius

        image_file = pair_images[pair]

        with Image.open(image_file) as opened_image:
            image = opened_image.convert("RGBA").copy()

        imagebox = offsetbox.OffsetImage(
            image,
            zoom=zoom,
        )

        annotation = offsetbox.AnnotationBbox(
            imagebox,
            (x_img, y_img),
            frameon=False,
        )

        ax.add_artist(annotation)

    # ------------------------------------------------------------
    # 9. Draw nodes and pair occupations
    # ------------------------------------------------------------

    for i, pair in enumerate(pairs):
        p, pbar = pair
        occupation = orbital_occupations[i]

        ax.text(
            x_coords[i] * 1.5,
            y_coords[i] * 1.5,
            f"{occupation:.2f} ({p},{pbar})",
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
        )

        circle = plt.Circle(
            (x_coords[i], y_coords[i]),
            radius=0.05,
            color="tab:blue",
            alpha=1.0,
            zorder=3,
        )

        ax.add_artist(circle)

    # ------------------------------------------------------------
    # 10. Draw pair mutual-correlation connections
    # ------------------------------------------------------------

    for i in range(num_pairs):
        for j in range(i + 1, num_pairs):
            value = mutual_correlation_matrix[i, j]

            plot_smooth_connection(
                ax,
                x_coords,
                y_coords,
                i,
                j,
                value,
                vmin,
                vmax,
                cmap,
            )

    # ------------------------------------------------------------
    # 11. Figure formatting
    # ------------------------------------------------------------

    plot_limit = radius + offset + 0.6

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    norm = mcolors.LogNorm(
        vmin=vmin,
        vmax=vmax,
    )

    scalar_mappable = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm,
    )

    scalar_mappable.set_array([])

    if show_colorbar:
        colorbar = fig.colorbar(
            scalar_mappable,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            orientation="vertical",
        )

        colorbar.set_label(
            "Mutual Correlation",
            rotation=270,
            labelpad=15,
        )

    # ------------------------------------------------------------
    # 12. Save and display
    # ------------------------------------------------------------

    if output_file is not None:
        logging.getLogger(
            "fontTools"
        ).setLevel(logging.WARNING)

        output_path = FilePath(output_file)

        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")

        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()



def mutual_correlation_compare_plot(
    system,
    C,
    indices,
    mutual_correlation_matrix_a,
    mutual_correlation_matrix_b,
    orbital_occupations=None,
    rotation_matrix_a=None,
    rotation_matrix_b=None,
    orbitals_filepath="mca_orbitals",
    radius=1.0,
    offset=1.5,
    zoom=0.2,
    fontsize=10,
    figsize=(6, 6),
    output_file=None,
    vmin=0.00075,
    vmax=0.75,
    cmap_name="coolwarm",
    show_colorbar=True,
    vmd_parameters=None,
    label_format="{:+.3f}",
    annotate_threshold=1e-6,
):
    """
    Compare two mutual-correlation matrices on the same orbital network.

    The edge color and width reflect the signed difference
    ``M2_b - M2_a``. Edge labels show the numeric difference directly,
    which is useful for visualizing rel vs nonrel changes.

    Parameters
    ----------
    system : System
        The Forte2 System object.
    C : NDArray
        The molecular orbital coefficients.
    indices : List[int]
        List of orbital indices to plot.
    mutual_correlation_matrix_a : NDArray
        First mutual-correlation matrix, e.g. nonrel M2.
    mutual_correlation_matrix_b : NDArray
        Second mutual-correlation matrix, e.g. rel M2.
    orbital_occupations : NDArray, optional
        Orbital occupations used for the orbital labels. If None, labels are omitted.
    rotation_matrix_a : NDArray, optional
        Optional rotation matrix applied to the first matrix before comparison.
    rotation_matrix_b : NDArray, optional
        Optional rotation matrix applied to the second matrix before comparison.
    orbitals_filepath : str, optional, default="mca_orbitals"
        Directory to save orbital cube files.
    radius : float, optional, default=1.0
        Radius of the circle on which orbitals are placed.
    offset : float, optional, default=1.5
        Offset for placing orbital images.
    zoom : float, optional, default=0.2
        Zoom factor for orbital images.
    fontsize : int, optional, default=10
        Font size for labels.
    figsize : Tuple[float, float], optional, default=(6, 6)
        Size of the figure.
    output_file : str, optional
        If provided, saves the plot to a file with this name (PDF format).
    vmin : float, optional, default=0.00075
        Minimum value for color mapping.
    vmax : float, optional, default=0.75
        Maximum value for color mapping.
    cmap_name : str, optional, default="coolwarm"
        Name of the matplotlib colormap to use.
    show_colorbar : bool, optional, default=True
        Whether to display the colorbar.
    vmd_parameters : dict, optional
        Parameters to pass to VMDCube for orbital visualization.
    label_format : str, optional
        Format string for the edge value labels.
    annotate_threshold : float, optional
        Only annotate edges with absolute difference above this threshold.
    """

    m2_a = np.asarray(mutual_correlation_matrix_a, dtype=float)
    m2_b = np.asarray(mutual_correlation_matrix_b, dtype=float)

    num_orbitals = len(indices)
    if m2_a.shape != (num_orbitals, num_orbitals):
        raise ValueError(
            f"mutual_correlation_matrix_a must have shape {(num_orbitals, num_orbitals)}, got {m2_a.shape}."
        )
    if m2_b.shape != (num_orbitals, num_orbitals):
        raise ValueError(
            f"mutual_correlation_matrix_b must have shape {(num_orbitals, num_orbitals)}, got {m2_b.shape}."
        )

    if rotation_matrix_a is not None:
        rotation_matrix_a = np.asarray(rotation_matrix_a, dtype=float)
        if rotation_matrix_a.shape != (num_orbitals, num_orbitals):
            raise ValueError(
                f"rotation_matrix_a must have shape {(num_orbitals, num_orbitals)}, got {rotation_matrix_a.shape}."
            )
        m2_a = rotation_matrix_a.T @ m2_a @ rotation_matrix_a

    if rotation_matrix_b is not None:
        rotation_matrix_b = np.asarray(rotation_matrix_b, dtype=float)
        if rotation_matrix_b.shape != (num_orbitals, num_orbitals):
            raise ValueError(
                f"rotation_matrix_b must have shape {(num_orbitals, num_orbitals)}, got {rotation_matrix_b.shape}."
            )
        m2_b = rotation_matrix_b.T @ m2_b @ rotation_matrix_b

    delta = m2_b - m2_a

    write_orbital_cubes(
        system, C, indices=indices, filepath=orbitals_filepath, prefix="orbital"
    )

    try:
        from vmdcube import VMDCube
    except ImportError:
        raise ImportError("VMDCube is not installed")

    vmd_parameters = {} if vmd_parameters is None else vmd_parameters
    vmd = VMDCube(cubedir=orbitals_filepath, **vmd_parameters)
    vmd.run()

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    cmap = colormaps[cmap_name]
    angles = np.linspace(0, 2 * np.pi, num_orbitals, endpoint=False)
    x_coords = radius * np.sin(angles)
    y_coords = radius * np.cos(angles)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")

    orbitals_filepath = FilePath(orbitals_filepath)
    tga_files = glob.glob(str(orbitals_filepath / FilePath("orbital_*.tga")))
    tga_files_dict = {}
    for file in tga_files:
        name = FilePath(file).stem
        if not cube_nonzero(f"{orbitals_filepath}/{name}.cube"):
            print(name, "is zero, skipping image.")
            continue
        import re
        match = re.match(r"orbital_(\d+)(?:_([ab]))?$", name)
        if match:
            orbital_index = int(match.group(1))
            tga_files_dict[orbital_index] = file

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        orbital_index = indices[i]
        x_img = (radius + offset) * x / radius
        y_img = (radius + offset) * y / radius
        filename = tga_files_dict[orbital_index]
        try:
            img = Image.open(filename)
            imagebox = offsetbox.OffsetImage(img, zoom=zoom)
            ab = offsetbox.AnnotationBbox(
                imagebox,
                (x_img, y_img),
                frameon=False,
            )
            ax.add_artist(ab)
        except FileNotFoundError:
            print(f"Warning: Could not find file {filename}")

    if orbital_occupations is not None:
        orbital_occupations = np.asarray(orbital_occupations, dtype=float)
        if orbital_occupations.shape != (num_orbitals,):
            raise ValueError(
                f"orbital_occupations must have shape ({num_orbitals},), got {orbital_occupations.shape}."
            )
        for i in range(num_orbitals):
            ax.text(
                x_coords[i] * 1.5,
                y_coords[i] * 1.5,
                f"{orbital_occupations[i]:.2f} ({indices[i]})",
                ha="center",
                va="center",
                fontsize=fontsize,
            )

    for i in range(num_orbitals):
        for j in range(i + 1, num_orbitals):
            val = delta[i, j]
            if abs(val) < annotate_threshold:
                continue

            color = cmap(0.5 + 0.5 * np.tanh(val / max(abs(vmax), 1e-12)))
            lw = 1.0 + 3.0 * min(abs(val) / max(abs(vmax), 1e-12), 1.0)

            p0 = [x_coords[i], y_coords[i]]
            p1 = [0.1 * (x_coords[i] + x_coords[j]), 0.1 * (y_coords[i] + y_coords[j])]
            p2 = [x_coords[j], y_coords[j]]
            path = MplPath(
                [p0, p1, p2],
                [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
            )
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                lw=lw,
                alpha=0.85,
            )
            ax.add_patch(patch)

            mid_x = 0.5 * (x_coords[i] + x_coords[j]) + 0.08 * (y_coords[j] - y_coords[i])
            mid_y = 0.5 * (y_coords[i] + y_coords[j]) - 0.08 * (x_coords[j] - x_coords[i])
            ax.text(
                mid_x,
                mid_y,
                label_format.format(val),
                fontsize=max(6, fontsize - 2),
                color=color,
                ha="center",
                va="center",
            )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    norm = mcolors.Normalize(vmin=-abs(vmax), vmax=abs(vmax))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if show_colorbar:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, orientation="vertical")
        cbar.set_label("ΔM2 (B - A)", rotation=270, labelpad=15)

    if output_file:
        import logging
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        plt.savefig(f"{output_file}.pdf", bbox_inches="tight")
    plt.show()


def read_cube_values(path):
    path = FilePath(path)

    with path.open() as f:
        lines = f.readlines()

    # Cube format:
    # line 0-1: comments
    # line 2: natoms + origin
    # line 3-5: grid
    # next natoms lines: atoms
    natoms = abs(int(lines[2].split()[0]))
    data_start = 2 + 1 + 3 + natoms

    values = np.fromstring(" ".join(lines[data_start:]), sep=" ")
    return values

def cube_nonzero(path, tol=1e-12):
    values = read_cube_values(path)
    max_abs = np.max(np.abs(values))
    nnz = np.count_nonzero(np.abs(values) > tol)

    return max_abs > tol


import numpy as np


def pair_mutual_correlation(lam2, P_pair, Q_pair, return_terms=False):
    """
    Mutual correlation between two Kramers-pair subspaces.

    Parameters
    ----------
    lam2 : ndarray, shape (nspinor, nspinor, nspinor, nspinor)
        Two-body cumulant with convention

            lam2[p, q, r, s] = lambda_{rs}^{pq}

    P_pair : tuple[int, int]
        (p, p_bar)

    Q_pair : tuple[int, int]
        (q, q_bar)

    return_terms : bool
        If True, also return the four contributions.

    Returns
    -------
    Mab : float
        Mutual correlation between P_pair and Q_pair.
    """

    lam2 = np.asarray(lam2)

    A = np.asarray(P_pair, dtype=int)
    B = np.asarray(Q_pair, dtype=int)

    if lam2.ndim != 4:
        raise ValueError("lam2 must be a rank-4 tensor.")

    if len(A) != 2 or len(B) != 2:
        raise ValueError("Each Kramers pair must contain exactly two spinors.")

    if len(np.unique(A)) != 2 or len(np.unique(B)) != 2:
        raise ValueError("The two members of each pair must be different.")

    if np.intersect1d(A, B).size != 0:
        raise ValueError("P_pair and Q_pair must be disjoint.")

    if np.any(A < 0) or np.any(B < 0):
        raise IndexError("Spinor indices must be nonnegative.")

    if np.any(A >= lam2.shape[0]) or np.any(B >= lam2.shape[0]):
        raise IndexError("A pair index is outside the lam2 spinor space.")

    # a in A; b,c,d in B
    block1 = lam2[np.ix_(A, B, B, B)]
    term1 = np.sum(np.abs(block1) ** 2)

    # a,b in A; c,d in B
    block2 = lam2[np.ix_(A, A, B, B)]
    term2 = 0.5 * np.sum(np.abs(block2) ** 2)

    # a,c in A; b,d in B
    block3 = lam2[np.ix_(A, B, A, B)]
    term3 = np.sum(np.abs(block3) ** 2)

    # a,b,c in A; d in B
    block4 = lam2[np.ix_(A, A, A, B)]
    term4 = np.sum(np.abs(block4) ** 2)

    Mab = np.real_if_close(term1 + term2 + term3 + term4).item()

    if return_terms:
        terms = {
            "A_BBB": np.real_if_close(term1).item(),
            "AA_BB": np.real_if_close(term2).item(),
            "A_B_A_B": np.real_if_close(term3).item(),
            "AAA_B": np.real_if_close(term4).item(),
        }
        return Mab, terms

    return Mab
