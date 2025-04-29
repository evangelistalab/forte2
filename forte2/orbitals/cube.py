import math
import numpy as np

import forte2
from forte2.orbitals.extents import orbital_extents


def local_grid(basis, C):
    coords, moments = orbital_extents(basis, C)
    extent_axis = []
    for i in range(C.shape[1]):
        # build a 3 x 3 tensor from the moments
        xx, xy, xz, yy, yz, zz = moments[i]
        M = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

        # diagonalize the tensor
        extent_axis.append(np.linalg.eigh(M))
    return coords, extent_axis


def simple_grid(atoms, overage=[5.0, 5.0, 5.0], spacing=[0.2, 0.2, 0.2]):
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
    print(xrange, yrange, zrange)
    npoints = []
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

    def __init__(self):
        pass

    def run(self, system, C):
        # determine the grid points for the cube file
        minr, npoints, axis = simple_grid(system.atoms)

        coords, extent_axis = local_grid(system.basis, C)

        # evaluate the orbital at the grid points
        values = forte2.ints.orbitals_on_grid(
            system.basis,
            C,
            minr,
            npoints,
            axis,
        )

        # write the cube files
        for i in range(C.shape[1]):
            cube = self.make_cube(values[:, i], minr, npoints, axis, system)
            with open(f"orbital_{i}.cube", "w") as f:
                f.write(cube)

    def make_cube(self, values, minr, npoints, axis, system):
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
            f"{Z:6d} {0.0:10.6f} {x:10.6f} {y:10.6f} {z:10.6f}"
            for Z, (x, y, z) in system.atoms
        )

        v = values.flatten()
        lines = [
            " ".join(f"{x:.5E}" for x in v[i : i + 6]) for i in range(0, len(v), 6)
        ]
        return header + "\n" + atoms + "\n" + "\n".join(lines)
