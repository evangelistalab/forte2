import glob
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps, offsetbox
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image
from forte2.orbitals import write_orbital_cubes

try:
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
except Exception:
    pass

def get_color_and_alpha_smooth(
    value, vmin, vmax, cmap, signed=False, signed_linthresh=None):
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
    signed : Bool
        Creates a diverging scale centered at zero instead of using magnitudes. 
    signed_linthresh : float
        Near-zero threshold value for the signed scale.

    Returns
    -------
    color : tuple
        RGBA tuple from the colormap.
    alpha : float
        Transparency in [0, 1], scaled logarithmically.
    """
    
    value = float(value)
    if signed:
        max_abs = max(abs(float(vmin)), abs(float(vmax)))
        if max_abs == 0.0:
            return cmap(0.5), 0.0

        linthresh = (
            max_abs * 1.0e-3 if signed_linthresh is None else float(signed_linthresh)
        )
        linthresh = min(max(linthresh, np.finfo(float).tiny), max_abs)

        color_norm = mcolors.SymLogNorm(
            linthresh=linthresh,
            vmin=-max_abs,
            vmax=max_abs,
            base=10,
        )
        clipped_value = float(np.clip(value, -max_abs, max_abs))
        color = cmap(color_norm(clipped_value))

        magnitude = abs(clipped_value)
        if magnitude == 0.0:
            alpha = 0.0
        else:
            alpha_norm = mcolors.LogNorm(vmin=linthresh, vmax=max_abs)
            alpha = float(alpha_norm(np.clip(magnitude, linthresh, max_abs)))
        return color, alpha

    value = float(np.clip(value, vmin, vmax))
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    color = cmap(norm(value))
    alpha = float(norm(value))
    return color, alpha

def plot_smooth_connection(
    ax,
    x_coords,
    y_coords,
    i,
    j,
    val,
    vmin,
    vmax,
    cmap="seismic",
    signed=False,
    signed_linthresh=None,
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
    color, alpha = get_color_and_alpha_smooth(
        val,
        vmin,
        vmax,
        cmap,
        signed=signed,
        signed_linthresh=signed_linthresh,
    )
    # Define the three points
    p0 = [x_coords[i], y_coords[i]]
    p1 = [
        0.1 * (x_coords[i] + x_coords[j]),
        0.1 * (y_coords[i] + y_coords[j]),
    ]
    p2 = [x_coords[j], y_coords[j]]
    
    # Create a Path for a quadratic Bezier curve
    path = Path([p0, p1, p2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    patch = PathPatch(
        path, facecolor="none", edgecolor=color, lw=1 + 3 * alpha, alpha=alpha
    )
    ax.add_patch(patch)

def _read_cube_data(cube_file):
    """Read enough of a Gaussian cube file to recover the volumetric data."""
    with open(cube_file, "r") as f:
        f.readline()
        f.readline()

        natoms = abs(int(f.readline().split()[0]))
        shape = []
        for _ in range(3):
            shape.append(abs(int(f.readline().split()[0])))

        for _ in range(natoms):
            f.readline()

        values = np.fromiter(
            (float(x) for line in f for x in line.split()), dtype=float
        )

    expected = int(np.prod(shape))
    if values.size < expected:
        raise ValueError(
            f"Cube file {cube_file} has {values.size} data values, expected {expected}"
        )
    return values[:expected].reshape(shape)

def _signed_max_projection(data, axis=2):
    """Project a signed orbital volume while preserving the larger-magnitude sign."""
    axis = int(axis)
    max_idx = np.argmax(np.abs(data), axis=axis)
    return np.take_along_axis(data, np.expand_dims(max_idx, axis), axis=axis).squeeze(
        axis
    )

def _render_cube_thumbnail(
    cube_file,
    image_file,
    projection_axis=0,
    cmap="seismic",
    dpi=300,
    levels=10,
):
    data = _read_cube_data(cube_file)
    plane = _signed_max_projection(data, axis=projection_axis)

    vmax = float(np.max(np.abs(plane)))
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(1.6, 1.6), dpi=dpi)
    ax.imshow(
        plane.T,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )

    contour_levels = np.linspace(-vmax, vmax, 2 * levels + 1)
    contour_levels = contour_levels[contour_levels != 0.0]
    if contour_levels.size:
        ax.contour(
            plane.T,
            levels=contour_levels,
            colors="black",
            linewidths=0.25,
            alpha=0.35,
        )
    ax.set_axis_off()
    fig.savefig(image_file, transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

def _render_orbital_images_from_cubes(
    orbitals_filepath,
    projection_axis=0,
    cmap="seismic",
    dpi=200,
    levels=10,
):
    orbitals_filepath = pathlib.Path(orbitals_filepath)
    image_files = {}
    cube_files = glob.glob(str(orbitals_filepath / "orbital_*.cube"))

    for cube_file in cube_files:
        cube_path = pathlib.Path(cube_file)
        orbital_index = int(cube_path.stem.split("_")[-1])
        image_file = orbitals_filepath / f"{cube_path.stem}.png"
        _render_cube_thumbnail(
            cube_path,
            image_file,
            projection_axis=projection_axis,
            cmap=cmap,
            dpi=dpi,
            levels=levels,
        )
        image_files[orbital_index] = image_file

    return image_files

#updated compatibility with mutual_correlation_mod and mod_fast, which have M2 dict objects
def _correlation_matrix(mca):
    if hasattr(mca, "get_M2_matrix"):
        return mca.get_M2_matrix()
    return np.asarray(mca)

#use user supplied coefficients first; if none, use nat orbs if available (mca nat_orbs=True)
def _plot_coefficients(C, mca):
    if C is not None:
        return C

    if getattr(mca, "nat_orbs", False) and getattr(mca, "C", None) is not None:
        return mca.C

    raise ValueError("No orbital coefficients available.")

#a smoke check that codex wrote-- necessary?
def _selected_occupation_numbers(occupation_numbers, indices, num_orbitals):
    occupation_numbers = np.asarray(occupation_numbers, dtype=float).reshape(-1)
    if occupation_numbers.shape[0] == num_orbitals:
        return occupation_numbers
    if occupation_numbers.shape[0] > max(indices):
        return occupation_numbers[np.asarray(indices, dtype=int)]
    raise ValueError(
        "occupation_numbers must either match the number of plotted orbitals "
        "or be indexable by the provided orbital indices."
    )

def _mca_occupation_numbers(mca):
    #uses natural orbital occupations if available; else, fall back to gamma1
    #if occupation_numbers is specified (eg. = ci.nat_occs) then this function is not called
    no_occupations = getattr(mca, "no_occupations", None)
    if no_occupations is not None:
        return np.asarray(no_occupations, dtype=float).reshape(-1)

    gamma1 = getattr(mca, "gamma1", None)
    if gamma1 is not None:
        gamma1 = np.asarray(gamma1)
        return np.diag(gamma1)

    return None

def _occupation_labels(mca, indices, num_orbitals, occupation_numbers=None):
    if getattr(mca, "nat_orbs", False):
        mca_occupations = _mca_occupation_numbers(mca)
        if mca_occupations is not None:
            return [
                f"{mca_occupations[i]:.3f} ({indices[i]})"
                for i in range(num_orbitals)
            ]

    if occupation_numbers is not None:
        occupation_numbers = _selected_occupation_numbers(
            occupation_numbers, indices, num_orbitals
        )
        return [
            f"{occupation_numbers[i]:.3f} ({indices[i]})"
            for i in range(num_orbitals)
        ]

    mca_occupations = _mca_occupation_numbers(mca)
    if mca_occupations is not None:
        return [
            f"{mca_occupations[i]:.3f} ({indices[i]})"
            for i in range(num_orbitals)
        ]

    return [f"{indices[i]}" for i in range(num_orbitals)]

def mutual_correlation_plot(
    system,
    C,
    indices,
    mca,
    title,
    orbitals_filepath="mca_orbitals",
    radius=1.0,
    offset=1.5,
    zoom=0.2,
    fontsize=10,
    figsize=(6, 6),
    output_file=None,
    vmin=1e-3,
    vmax=1,
    cmap_name="seismic",
    show_colorbar=True,
    projection_axis=0,
    signed_correlation=None,
    signed_linthresh=None,
    occupation_numbers=None,
):
    """
    Plots a set of orbitals arranged in a circle, and visualizes diagonal, semi-diagonal,
    and off-diagonal terms of the 2-body reduced density cumulant.

    Parameters
    ----------
    system : System
        The Forte2 System object.
    C : ndarray or None
        Molecular orbital coefficients. If None and the MCA object was
        constructed with nat_orbs=True, the stored orbital coefficients
        are used automatically.
    indices : List[int]
        List of orbital indices to plot.
    mca : MutualCorrelationAnalysis
        The MutualCorrelationAnalysis object containing the cumulant data.
    title : str
        Writes a title on the plot
    orbitals_filepath : str, optional, default="mca_orbitals"
        Directory to save orbital cube files.
    radius : float, optional, default=1.5
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
    vmin, vmax : float, optional, default=10
        Minimum, maximum value for color mapping.
    cmap_name : str, optional, default="magma_r"
        Name of the matplotlib colormap to use.
    show_colorbar : bool, optional, default=True
        Whether to display the colorbar.
    projection_axis : int; 0 -> x, 1 -> y, 2 -> z.
        The axis on which to splice cube files for 2-D rendering.
    signed_correlation : int, optional
    ..
    signed_linthresh : int, optional
    ..
    occupation_numbers : array-like, optional
        Occupation numbers to display. If omitted, the function first uses
        natural orbital occupations stored in the MCA object, then
        falls back to the diagonal of the stored one-particle RDM.
    """

    C = _plot_coefficients(C, mca)
    write_orbital_cubes(
        system, C, indices=indices, filepath=orbitals_filepath, prefix="orbital"
    )

    orbital_render_options = {} 
    orbital_render_options.setdefault("projection_axis", projection_axis)
    image_files_dict = _render_orbital_images_from_cubes(
        orbitals_filepath, **orbital_render_options
    )

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    # if mca.nfragments != len(indices):
    #     raise ValueError(
    #         "Plotting currently requires one orbital per fragment."
    #     )
    num_orbitals = len(indices)

    correlation_values = _correlation_matrix(mca)

    if signed_correlation is None:
        signed_correlation = True

    #if signed_correlation and cmap_name == "magma_r":
    #    cmap_name = "coolwarm"

    cmap = colormaps[cmap_name]

    angles = np.linspace(0, 2 * np.pi, num_orbitals, endpoint=False)
    x_coords = radius * np.sin(angles)
    y_coords = radius * np.cos(angles)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        orbital_index = indices[i]

        x_img = (radius + offset) * x / radius
        y_img = (radius + offset) * y / radius

        image_file = image_files_dict.get(orbital_index)
        if image_file is None:
            print(f"Warning: Could not find rendered image for orbital {orbital_index}")
            continue

        img = Image.open(image_file)
        imagebox = offsetbox.OffsetImage(img, zoom=zoom)
        ab = offsetbox.AnnotationBbox(imagebox, (x_img, y_img), frameon=False)
        ax.add_artist(ab)

    if correlation_values.shape[0] != num_orbitals:
        raise ValueError(
            "The number of orbitals used in the mutual correlation data "
            "does not match the number of orbitals provided."
        )

    labels = _occupation_labels(
        mca,
        indices,
        num_orbitals,
        occupation_numbers=occupation_numbers,
    )
    for i in range(num_orbitals):
        ax.text(
            x_coords[i] * 1.5,
            y_coords[i] * 1.5,
            labels[i],
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        circle = plt.Circle((x_coords[i], y_coords[i]), 0.05, alpha=1.0, zorder=2)
        ax.add_artist(circle)

    for i in range(num_orbitals):
        for j in range(i + 1, num_orbitals):
            val = correlation_values[i, j]
            plot_smooth_connection(
                ax,
                x_coords,
                y_coords,
                i,
                j,
                val,
                vmin,
                vmax,
                cmap,
                signed=signed_correlation,
                signed_linthresh=signed_linthresh,
            )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_title(title)

    if signed_correlation:
        max_abs = max(abs(float(vmin)), abs(float(vmax)))
        if max_abs == 0.0:
            max_abs = 1.0
        linthresh = (
            max_abs * 1.0e-3 if signed_linthresh is None else float(signed_linthresh)
        )
        linthresh = min(max(linthresh, np.finfo(float).tiny), max_abs)
        norm = mcolors.SymLogNorm(
            linthresh=linthresh,
            vmin=-max_abs,
            vmax=max_abs,
            base=10,
        )
    else:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if show_colorbar:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, orientation="vertical")
        cbar.set_label("Mutual Correlation Energy", rotation=270, labelpad=15)

    if output_file:
        import logging

        logging.getLogger("fontTools").setLevel(logging.WARNING)
        output_path = pathlib.Path(output_file)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        fig.savefig(output_path, bbox_inches="tight")
    
    plt.show()
