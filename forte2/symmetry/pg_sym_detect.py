from dataclasses import dataclass
import itertools
import numpy as np
import scipy as sp

from forte2.helpers import logger
from .sym_utils import rotation_mat, reflection_mat, equivalent_under_operation


def _is_colinear(v1, v2, tol=1e-4):
    cross_prod = np.cross(v1, v2)
    return np.linalg.norm(cross_prod) < tol


@dataclass
class PGSymmetryDetector:
    """
    Class to detect Abelian point group symmetry of a molecule.

    Parameters
    ----------
    inertia_tensor : ndarray of shape (3, 3)
        Moment of inertia tensor.
    com_atomic_positions : ndarray of shape (natoms, 3)
        Atomic positions in the center-of-mass frame.
    charges : ndarray of shape (natoms,)
        Atomic numbers.
    tol : float, optional, default=1e-4
        Tolerance for detecting symmetry.

    Notes
    -----
    See https://github.com/NASymmetry/MolSym/blob/main/molsym/pgdetect/flowchart.py
    We first find the principal axes of rotation, and then determine
    the largest Abelian point group.

    To find the principal axes, we first compute the principal
    moments of inertia and their corresponding axes.
    Depending on the number of degenerate moments of inertia,
    the molecule is classified as an asymmetric top (non-degenerate),
    symmetric top (doubly degenerate), or spherical top (triply degenerate).

    The asymmetric top case is the simplest: these correspond to one of the subgroups of D2h,
    and the principal axes are the eigenvectors of the inertia tensor.

    The symmetric top case has a unique axis (e.g. the lone pair axis in NH3), and the other two axes
    are arbitrary in the plane orthogonal to the unique axis.
    Symmetry equivalent atoms are found, and any atom with a non-zero projection onto the orthogonal plane is used to define the x-axis.
    The y-axis is then defined as the cross product of the unique axis and x-axis.

    The spherical top case is the most complicated. To distinguish between T/O/I groups, we find all unique C2 axes.
    T groups have 3 unique C2 axes, O groups have 9, and I groups have 15.
    For T groups, the principal axes are just the C2 axes.
    For O groups, there will be 3 unique C4 axes, which will be the principal axes.
    I groups are currently treated as C1.
    """

    inertia_tensor: np.ndarray
    com_atomic_positions: np.ndarray
    charges: np.ndarray
    tol: float = 1e-4

    def run(self):
        self.natoms = self.com_atomic_positions.shape[0]
        if self.natoms == 1:
            # Just an atom at the origin, no need to rotate
            self.prinrot = np.eye(3)
            self.prin_atomic_positions = self.com_atomic_positions
            self.pg_name = "D2H"
            return

        # compute principal moments of inertia. These are sorted in ascending order
        self.moi, self.moi_vectors = np.linalg.eigh(self.inertia_tensor)
        logger.log_info1(f"Principal moments of inertia: {self.moi}")

        self.find_symmetry_equivalent_atoms()

        # count degeneracies
        ndegen = (np.abs(self.moi[1:] - self.moi[:-1]) < self.tol).sum() + 1

        force_c1 = False
        if ndegen == 1:
            self.prinrot = self.find_principal_rotation_axes_asym_top()
        elif ndegen == 2:
            self.prinrot, force_c1 = self.find_principal_rotation_axes_sym_top()
        else:
            self.prinrot, force_c1 = self.find_principal_rotation_axes_sph_top()

        det = np.linalg.det(self.prinrot)

        if not np.isclose(np.abs(det), 1.0, atol=self.tol):
            # re-orthogonalize
            u, _, vh = np.linalg.svd(self.prinrot)
            self.prinrot = u @ vh
            det = np.linalg.det(self.prinrot)

        if det < 0:
            # make it a proper rotation (det=1)
            self.prinrot[2, :] *= -1

        self.prin_atomic_positions = (self.prinrot @ self.com_atomic_positions.T).T

        if force_c1:
            self.pg_name = "C1"
        else:
            self.pg_name = self.detect_abelian_pg_symmetry()

    def detect_abelian_pg_symmetry(self):
        """
        After determining the principal axes of rotation,
        this method determine the largest Abelian point group symmetry of the molecule.
        """

        # 1. Check for inversion center
        # every atom must have a partner of the same type at -R
        has_inversion = equivalent_under_operation(
            self.prin_atomic_positions, self.charges, lambda R: -R, self.tol
        )

        # 2. Check for C2 axes
        has_C2x = equivalent_under_operation(
            self.prin_atomic_positions,
            self.charges,
            lambda R: rotation_mat((1, 0, 0), np.deg2rad(180.0)) @ R,
            self.tol,
        )
        has_C2y = equivalent_under_operation(
            self.prin_atomic_positions,
            self.charges,
            lambda R: rotation_mat((0, 1, 0), np.deg2rad(180.0)) @ R,
            self.tol,
        )
        has_C2z = equivalent_under_operation(
            self.prin_atomic_positions,
            self.charges,
            lambda R: rotation_mat((0, 0, 1), np.deg2rad(180.0)) @ R,
            self.tol,
        )
        nC2 = sum([has_C2x, has_C2y, has_C2z])
        assert nC2 in [0, 1, 3], f"Found {nC2} C2 axes, which is unexpected."

        # 3. Check for mirror planes
        has_Sxy = equivalent_under_operation(
            self.prin_atomic_positions,
            self.charges,
            lambda R: reflection_mat((0, 1)) @ R,
            self.tol,
        )
        has_Sxz = equivalent_under_operation(
            self.prin_atomic_positions,
            self.charges,
            lambda R: reflection_mat((0, 2)) @ R,
            self.tol,
        )
        has_Syz = equivalent_under_operation(
            self.prin_atomic_positions,
            self.charges,
            lambda R: reflection_mat((1, 2)) @ R,
            self.tol,
        )
        nSigma = sum([has_Sxy, has_Sxz, has_Syz])

        # 4. Determine point group
        # {(has_inversion, nC2, nSigma): pg_name}
        pg_dict = {
            (False, 0, 0): "C1",
            (True, 0, 0): "CI",
            (False, 1, 0): "C2",
            (False, 0, 1): "CS",
            (False, 3, 0): "D2",
            (False, 1, 2): "C2V",
            (True, 1, 1): "C2H",
            (True, 3, 3): "D2H",
        }
        try:
            pg = pg_dict[(has_inversion, nC2, nSigma)]
        except KeyError:
            logger.log_warning(
                f"symmetry.py::detect_pg_symmetry: "
                f"Could not determine point group for has_inversion={has_inversion}, nC2={nC2}, nSigma={nSigma}. Setting to C1."
            )
            pg = "C1"
        return pg

    def find_principal_rotation_axes_asym_top(self):
        axis_order = []
        for i in range(3):
            R = rotation_mat(self.moi_vectors[:, i], np.pi)
            rotated_positions = (R @ self.com_atomic_positions.T).T
            all_match = True
            for x in self.com_atomic_positions:
                found = False
                for y in rotated_positions:
                    if np.linalg.norm(x - y) < self.tol:
                        found = True
                if not found:
                    all_match = False

            axis_order.append(2 if all_match else 1)

        sorted_axis = sorted(
            zip(axis_order, self.moi, range(3)),
            key=lambda x: (
                x[0],
                -x[1],
            ),  # sort axes by Cn order first, then by descending MOI
        )
        logger.log_debug("Sorted Axis Order:")
        for ax in sorted_axis:
            n, I, idx = ax
            logger.log_debug(
                f"Axis: {self.moi_vectors[:, idx]}   Cn: {n}   MOI: {I}   Axis Assignment: {idx}"
            )

        prinrot = self.moi_vectors[:, [n for _, _, n in sorted_axis]].T
        return prinrot

    def find_principal_rotation_axes_sym_top(self):
        force_c1 = False
        if abs(self.moi[0]) < self.tol:
            # linear molecule: arbitrary x/y plane is fine, don't bother with the rest
            prinrot = self.moi_vectors[:, [1, 2, 0]].T
        else:
            unique_axis = 0 if abs(self.moi[0] - self.moi[1]) > self.tol else 2
            z_axis = self.moi_vectors[:, unique_axis]
            # Find all possible C2 axes orthogonal to the unique axis
            c2_axes = []
            c2_axes += self.find_c2_axes_through_atom()
            c2_axes += self.find_c2_axes_through_midpoint()
            unique_c2_axes = [z_axis]
            for ax in c2_axes[1:]:
                is_unique = True
                for uax in unique_c2_axes:
                    if _is_colinear(ax, uax, tol=self.tol):
                        is_unique = False
                        break
                if is_unique:
                    unique_c2_axes.append(ax)
            if len(unique_c2_axes) == 1:
                # No C2 axes, but there could be mirror planes.
                # We only need to worry about Cnh and Cnv,
                # since Dnh and Dnd would have three unique C2 axes.
                # Cnh is easy, since we already know the unique axis,
                # any x/y axis will lie in the horizontal mirror plane.
                # For Cnv, we know the sigma_v plane must pass through
                # symmetry equivalent atoms, so pick one and we're done.
                for equiv_set in self.equivalent_sets:
                    for i in equiv_set:
                        vec = self.com_atomic_positions[i]
                        if _is_colinear(vec, z_axis, tol=self.tol):
                            continue
                        x_axis = vec - np.dot(vec, z_axis) * z_axis
                        x_axis /= np.linalg.norm(x_axis)
                        y_axis = np.cross(z_axis, x_axis)
                        break
                    break
            else:
                # found at least one C2 axis orthogonal to the unique axis
                # use the first one to define the x-axis
                x_axis = unique_c2_axes[1]
                y_axis = np.cross(z_axis, x_axis)
            prinrot = np.array([x_axis, y_axis, z_axis])
        return prinrot, force_c1

    def find_principal_rotation_axes_sph_top(self):
        force_c1 = False

        c2_axes = []
        c2_axes += self.find_c2_axes_through_atom()
        c2_axes += self.find_c2_axes_through_midpoint()
        unique_c2_axes = [c2_axes[0]]
        for ax in c2_axes[1:]:
            is_unique = True
            for uax in unique_c2_axes:
                if _is_colinear(ax, uax, tol=self.tol):
                    is_unique = False
                    break
            if is_unique:
                unique_c2_axes.append(ax)

        nc2 = len(unique_c2_axes)
        if nc2 not in [3, 9, 15]:
            logger.log_warning(
                f"find_principal_rotation_axes_sph_top: Found {nc2} unique C2 axes, which is unexpected."
                "Not reorienting. Check geometry, or relax tolerance."
            )
            prinrot = np.eye(3)
            force_c1 = True
        if nc2 == 3:
            # T/Td/Th, the C2 axes are the principal axes
            prinrot = np.array(unique_c2_axes)
        elif nc2 == 9:
            unique_c4_axes = self.find_c4_axes_perp_to_square()
            if len(unique_c4_axes) != 3:
                logger.log_warning(
                    f"find_principal_rotation_axes_sph_top: Octahedral symmetry detected,"
                    f" but found {len(unique_c4_axes)} unique C4 axes, which is unexpected."
                    " Not reorienting. Check geometry, or relax tolerance."
                )
                prinrot = np.eye(3)
                force_c1 = True
            else:
                prinrot = np.array(unique_c4_axes)
        elif nc2 == 15:
            logger.log_warning(
                "find_principal_rotation_axes_sph_top: Icosahedral point group detected, but currently treated as C1."
            )
            prinrot = np.eye(3)
            force_c1 = True

        return prinrot, force_c1

    def find_symmetry_equivalent_atoms(self):
        """
        Find sets of symmetry equivalent atoms based on interatomic distances.
        """
        distance_matrix = sp.spatial.distance.cdist(
            self.com_atomic_positions, self.com_atomic_positions, "euclidean"
        )
        natoms = self.com_atomic_positions.shape[0]
        self.equivalent_pairs = []
        for i in range(natoms):
            for j in range(i + 1, natoms):
                if self.charges[i] != self.charges[j]:
                    continue
                # if i and j are symmetry equivalent,
                # then they must have the same sorted distance list to all other atoms
                if np.allclose(
                    np.sort(distance_matrix[i, :].copy()),
                    np.sort(distance_matrix[j, :].copy()),
                    atol=self.tol,
                    rtol=0,
                ):
                    self.equivalent_pairs.append((i, j))

        self.equivalent_sets = []
        for i, j in self.equivalent_pairs:
            found = False
            for s in self.equivalent_sets:
                if i in s or j in s:
                    s.add(i)
                    s.add(j)
                    found = True
                    break
            if not found:
                self.equivalent_sets.append(set([i, j]))

    def find_c2_axes_through_atom(self):
        c2_axes_through_atom = []
        for equiv_set in self.equivalent_sets:
            for i in equiv_set:
                norm = np.linalg.norm(self.com_atomic_positions[i])
                # skip if atom is at origin (although the origin atom shouldn't be symmetry equivalent to any other atom)
                if norm < self.tol:
                    continue
                axis = self.com_atomic_positions[i] / norm
                if equivalent_under_operation(
                    self.com_atomic_positions,
                    self.charges,
                    lambda R: rotation_mat(axis, np.deg2rad(180.0)) @ R,
                    self.tol,
                ):
                    c2_axes_through_atom.append(axis)
        return c2_axes_through_atom

    def find_c2_axes_through_midpoint(self):
        c2_axes_through_midpoint = []
        for i, j in self.equivalent_pairs:
            mid = 0.5 * (self.com_atomic_positions[i] + self.com_atomic_positions[j])
            norm = np.linalg.norm(mid)
            # skip if midpoint is at origin (i.e., atoms are inversion partners)
            if norm < self.tol:
                continue
            axis = mid / norm
            if equivalent_under_operation(
                self.com_atomic_positions,
                self.charges,
                lambda R: rotation_mat(axis, np.deg2rad(180.0)) @ R,
                self.tol,
            ):
                c2_axes_through_midpoint.append(axis)
        return c2_axes_through_midpoint

    def find_c4_axes_perp_to_square(self):
        c4_axes = []
        # pick a set of symmetry equivalent atoms find all quadruplets that form a square
        # the normal of each square is a C4 axis
        equiv_set = self.equivalent_sets[0]
        assert (
            len(equiv_set) >= 6
        ), "Not enough symmetry equivalent atoms to define C4 axes."
        equiv_list = list(equiv_set)
        for quad in itertools.combinations(equiv_list, 4):
            pos = [self.com_atomic_positions[i] for i in quad]
            dists = sp.spatial.distance.pdist(pos, "euclidean")
            dists = np.sort(dists)
            # check if the 4 atoms form a square
            if (
                np.allclose(dists[0:4], dists[0], atol=self.tol, rtol=0)
                and np.allclose(dists[4:6], dists[4], atol=self.tol, rtol=0)
                and dists[4] > dists[0]
                and np.isclose(dists[4], np.sqrt(2) * dists[0], atol=self.tol, rtol=0)
            ):
                # normal of the square is a C4 axis
                v1 = pos[1] - pos[0]
                v2 = pos[2] - pos[0]
                axis = np.cross(v1, v2)
                axis /= np.linalg.norm(axis)
                c4_axes.append(axis)

        # keep only unique axes
        unique_c4_axes = [c4_axes[0]]
        for ax in c4_axes[1:]:
            is_unique = True
            for uax in unique_c4_axes:
                if _is_colinear(ax, uax, tol=self.tol):
                    is_unique = False
                    break
            if is_unique:
                unique_c4_axes.append(ax)
        return unique_c4_axes
