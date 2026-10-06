import numpy as np

from forte2 import System


def _detect(xyz, symmetry_tol=1e-4, unit="angstrom"):
    return System(
        xyz=xyz,
        basis_set="sto-6g",
        symmetry=True,
        symmetry_tol=symmetry_tol,
        unit=unit,
    )


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


def test_pg_detection_orients_cs_plane_as_xy():
    system = _detect("O 0 0 0; H 0.97 0 0; Cl -0.5 1.6 0")
    assert system.point_group == "CS"
    np.testing.assert_allclose(system.prin_atomic_positions[:, 2], 0, atol=1e-12)
