import numpy as np

from forte2 import System
from forte2.symmetry.mo_sym_detect import get_symmetry_ops


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


def test_pg_detection_honors_symmetry_tol():
    # One H is 3e-5 angstrom (5.7e-5 bohr) off the C2v geometry.
    water = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.58703"
    assert _detect(water, symmetry_tol=1e-6).point_group == "CS"
    assert _detect(water, symmetry_tol=1e-3).point_group == "C2V"


def test_pg_detection_near_symmetric_benzene():
    # Displacements of 1e-5 bohr, well within symmetry_tol, keep the full D2h group.
    rng = np.random.default_rng(265)
    displacements = rng.normal(size=(12, 3))
    displacements *= 1e-5 / np.linalg.norm(displacements, axis=1)[:, None]
    angles = np.arange(6) * np.pi / 3
    atoms = [("C", 2.63 * np.cos(a), 2.63 * np.sin(a)) for a in angles]
    atoms += [("H", 4.67 * np.cos(a), 4.67 * np.sin(a)) for a in angles]
    benzene = "\n".join(
        f"{el} {x + d[0]:.12f} {y + d[1]:.12f} {d[2]:.12f}"
        for (el, x, y), d in zip(atoms, displacements)
    )
    system = _detect(benzene, unit="bohr")
    assert system.point_group == "D2H"

    # Each operation maps the symmetrized atoms exactly onto their stored partners.
    positions = system.prin_atomic_positions
    for op, R in get_symmetry_ops(system.point_group).items():
        images = positions[system.atom_permutations[op]]
        np.testing.assert_allclose(positions @ R.T, images, atol=1e-12)


def test_pg_detection_accidental_symmetric_top():
    # Two equal moments of inertia make this water an accidental symmetric top whose
    # only C2 axis is perpendicular to the unique axis.
    water = "O 0 0 0; H 0 1.43 1.5174; H 0 -1.43 1.5174"
    assert _detect(water, unit="bohr").point_group == "C2V"
