import numpy as np

from forte2.system.geom_utils import parse_geometry


def test_parse_xyz_accepts_uppercase_two_letter_symbol():
    atoms = parse_geometry("NA 0.0 0.0 1.5", unit="bohr")

    assert len(atoms) == 1
    assert atoms[0][0] == 11
    np.testing.assert_allclose(atoms[0][1], [0.0, 0.0, 1.5])
