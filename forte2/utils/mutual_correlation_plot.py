import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib import colormaps
from matplotlib import offsetbox
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image
import pathlib

from forte2.orbitals import Cube

# retina display settings
try:
    get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
except Exception:
    pass


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

    # Create a Path for a quadratic Bezier curve
    verts = [p0, p1, p2]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

    path = Path(verts, codes)
    patch = PathPatch(
        path, facecolor="none", edgecolor=color, lw=1 + 3 * alpha, alpha=alpha
    )
    ax.add_patch(patch)


def mutual_correlation_plot(
    system,
    C,
    indices,
    mca,
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
    Plots a set of orbitals arranged in a circle, and visualizes diagonal, semi-diagonal,
    and off-diagonal terms of the 2-body reduced density cumulant.

    Parameters
    ----------
    system : System
        The Forte2 System object.
    C : NDArray
        The molecular orbital coefficients.
    indices : List[int]
        List of orbital indices to plot.
    mca : MutualCorrelationAnalysis
        The MutualCorrelationAnalysis object containing the cumulant data.
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
    cmap_name : str, optional, default="magma_r"
        Name of the matplotlib colormap to use.
    show_colorbar : bool, optional, default=True
        Whether to display the colorbar.
    vmd_parameters : dict, optional
        Parameters to pass to VMDCube for orbital visualization.
    """

    # generate cube files for the orbitals
    cube = Cube()
    cube.run(system, C, indices=indices, filepath=orbitals_filepath, prefix="orbital")

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

    num_orbitals = len(indices)

    # 1) Place orbitals on a circle
    angles = np.linspace(0, 2 * np.pi, num_orbitals, endpoint=False)
    x_coords = radius * np.sin(angles)
    y_coords = radius * np.cos(angles)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", "box")

    # find all the files with the pattern orbital_*.tga
    orbitals_filepath = pathlib.Path(orbitals_filepath)

    # form a dictionary mapping orbital index (int) to tga file path
    tga_files = glob.glob(str(orbitals_filepath / pathlib.Path("orbital_*.tga")))
    tga_files_dict = {}
    for file in tga_files:
        orbital_index = int(file.split("/")[-1].split(".")[0].split("_")[-1])
        tga_files_dict[orbital_index] = file

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
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

    if mca.M1.shape[0] != num_orbitals:
        raise ValueError(
            "The number of orbitals used in the MutualCorrelationAnalysis object does not match the number of orbitals provided."
        )

    # Label each orbital with the occupation number and index
    for i in range(num_orbitals):
        val = mca.Γ1[i, i]
        color, alpha = get_color_and_alpha_smooth(val, 0.01, 2, cmap)
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
            val = mca.M2[i, j]
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
        plt.savefig(f"{output_file}.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    from forte2.systems import System
    from forte2.solvers.hf import RHF
    from forte2.solvers.ci import CI
    from forte2.states import State
    from forte2.props.mutual_correlation import MutualCorrelationAnalysis

    xyz = f"""
    N 0.0 0.0 0.0
    N 0.0 0.0 1.1
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0), active_orbitals=list(range(10))
    )(rhf)
    ci.run()

    mca = MutualCorrelationAnalysis(ci)

    mutual_correlation_plot(
        system,
        ci.C[0],
        indices=ci.mo_space.active_indices,
        mca=mca,
        output_file="mutual_correlation_N2",
    )
