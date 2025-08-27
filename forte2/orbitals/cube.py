import math

from forte2 import ints


def simple_grid(
    atoms, spacing: tuple[float, float, float], overage: tuple[float, float, float]
):
    # find the orbital extents
    xrange = (math.inf, -math.inf)
    yrange = (math.inf, -math.inf)
    zrange = (math.inf, -math.inf)
    for _, (x, y, z) in atoms:
        xrange = (min(xrange[0], x), max(xrange[1], x))
        yrange = (min(yrange[0], y), max(yrange[1], y))
        zrange = (min(zrange[0], z), max(zrange[1], z))
    # add overage
    xrange = (xrange[0] - overage[0], xrange[1] + overage[0])
    yrange = (yrange[0] - overage[1], yrange[1] + overage[1])
    zrange = (zrange[0] - overage[2], zrange[1] + overage[2])
    npoints = [
        math.ceil((r[1] - r[0]) / s)
        for (r, s) in zip((xrange, yrange, zrange), spacing)
    ]
    return (
        (xrange[0], yrange[0], zrange[0]),
        tuple(npoints),
        [(spacing[0], 0, 0), (0, spacing[1], 0), (0, 0, spacing[2])],
    )


class Cube:
    """
    Class to handle cube files.
    """

    def __init__(self, spacing=0.2, overage=4.0):
        self.spacing = [spacing, spacing, spacing]
        self.overage = [overage, overage, overage]

    def run(self, system, C, indices=None, prefix="orbital"):
        # determine the indices of the orbitals to generate
        indices = indices if indices is not None else range(C.shape[1])
        max_index = max(indices)
        number_of_digits = int(math.log10(max_index + 1)) + 1

        # determine the grid points for the cube file
        grid_origin, npoints, scaled_axes = simple_grid(
            system.atoms, spacing=self.spacing, overage=self.overage
        )

        # calculate the orbitals on the grid
        values = ints.orbitals_on_grid(
            system.basis,
            C[:, indices],
            grid_origin,
            npoints,
            scaled_axes,
        )

        # write the cube files
        for i, index in enumerate(indices):
            cube = self._make_cube(
                values[:, i], grid_origin, npoints, scaled_axes, system
            )
            with open(f"{prefix}_{index + 1:0{number_of_digits}d}.cube", "w") as f:
                f.write(cube)

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
