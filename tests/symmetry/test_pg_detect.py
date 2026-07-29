import numpy as np
import pytest

from forte2 import System
from forte2.symmetry.pg_sym_detect import PGSymmetryDetector


def _bare_sym_top_detector():
    d = PGSymmetryDetector.__new__(PGSymmetryDetector)
    d.tol = 1e-4
    d.moi = np.array([1.0, 1.0, 2.0])  # doubly degenerate -> symmetric top, z unique
    d.moi_vectors = np.eye(3)  # z_axis = [0, 0, 1]
    return d


def test_sym_top_single_perpendicular_c2_axis_not_dropped():
    # Regression: the uniqueness loop iterated c2_axes[1:] while seeding
    # unique_c2_axes with z_axis (not one of c2_axes), silently dropping the
    # first detected perpendicular C2 axis. With exactly one such axis the code
    # then fell into the no-C2 branch and derived the x-axis from an atom
    # position instead of the true C2 axis.
    d = _bare_sym_top_detector()
    # atoms lie in the xy-plane at 45 deg, so the fallback (no-C2) branch would
    # give x ~ [0.707, 0.707, 0], distinct from the true C2 axis along x.
    d.com_atomic_positions = np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    d.charges = np.array([1, 1])
    d.equivalent_sets = [{0, 1}]
    d.find_c2_axes_through_atom = lambda: [np.array([1.0, 0.0, 0.0])]
    d.find_c2_axes_through_midpoint = lambda: []

    prinrot, _ = d._find_principal_rotation_axes_sym_top()
    # The x-axis must be the detected C2 axis, not the atom-derived direction.
    assert np.allclose(np.abs(prinrot[0]), [1.0, 0.0, 0.0])


def test_sym_top_cnv_fallback_on_axis_first_equiv_set():
    # Regression: the Cnv/Cnh fallback unconditionally broke after the first
    # equivalent set and only set x_axis if that set had an off-axis atom, so a
    # first set of purely on-axis atoms left x_axis unbound (UnboundLocalError).
    d = _bare_sym_top_detector()
    d.com_atomic_positions = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.3]]
    )
    d.charges = np.array([1, 1, 8])
    d.equivalent_sets = [{0, 1}, {2}]  # first set is on the z axis only
    d.find_c2_axes_through_atom = lambda: []
    d.find_c2_axes_through_midpoint = lambda: []

    # Must not raise; must return an orthonormal right-handed frame.
    prinrot, _ = d._find_principal_rotation_axes_sym_top()
    assert prinrot.shape == (3, 3)
    assert np.allclose(prinrot @ prinrot.T, np.eye(3), atol=1e-6)


def test_pg_detection_atom():
    xyz = """
    H 0 0 0
    """
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "d2h"


def test_pg_detection_ch4_with_zmat():
    xyz = """
    C
    H 1 1.2
    H 1 1.2 2 109.471221
    H 1 1.2 2 109.471221 3 120
    H 1 1.2 2 109.471221 3 -120
    """
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "d2"
