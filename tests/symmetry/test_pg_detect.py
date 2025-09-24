import numpy as np

from forte2 import System


def test_pg_detection_d2h():
    xyz = """
    C         -2.53339        0.70857       -0.33673
    C         -1.35409        0.67398        0.27977
    H         -2.86160       -0.13436       -0.93616
    H         -3.17344        1.57989       -0.24348
    H         -0.71405       -0.19735        0.18652
    H         -1.02588        1.51691        0.87920
    """
    system = System(xyz=xyz, basis_set="sto-3g", symmetry=True)
    assert system.point_group.lower() == "d2h"


def test_pg_detection_c3v():
    xyz = """
    C 0 0 0
    Cl 1 1 1
    H -1 -1 1
    H -1 1 -1
    H 1 -1 -1
    """
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "cs"


def test_pg_detection_td():
    xyz = """
    C         -1.81063        1.22554        0.21080
    H         -0.70123        1.22554        0.21080
    H         -2.18043        1.65730        1.16348
    H         -2.18043        0.18461        0.10838
    H         -2.18043        1.83470       -0.63945
    """
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "d2"


def test_pg_detection_d6h():
    xyz = """
    C       -0.7968025762      0.6132558281      1.4364406299                 
    C       -1.8139627858      1.5035441663      1.7842374187                 
    C       -1.6199615373      2.8776779426      1.6348699472                 
    C       -0.4088031047      3.3615229849      1.1376978877                 
    C        0.6083540562      2.4712342417      0.7898933093                 
    C        0.4143558370      1.0971008627      0.9392685784                 
    H        1.2047261985      0.4053142743      0.6690170614                 
    H       -0.9475465592     -0.4544960324      1.5525099643                 
    H       -2.7550789265      1.1275786411      2.1705536248                 
    H       -2.4103300853      3.5694647908      1.9051261025                 
    H       -0.2580573283      4.4292750974      1.0216331999                 
    H        1.5494665901      2.8471992552      0.4035678185                 
    """
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "d2h"


def test_pg_detection_oh_cubic():
    geom = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=float,
    )
    rand33 = np.random.rand(3, 3)
    u, _, vh = np.linalg.svd(rand33)
    random_rot = u @ vh
    if np.linalg.det(random_rot) < 0:
        random_rot[2, :] *= -1
    geom = (random_rot @ geom.T).T
    xyz = "\n".join([f"H {x[0]} {x[1]} {x[2]}" for x in geom])

    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "d2h"


def test_pg_detection_oh_octahedral():
    geom = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
            [0, 1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0],
        ],
        dtype=float,
    )
    rand33 = np.random.rand(3, 3)
    u, _, vh = np.linalg.svd(rand33)
    random_rot = u @ vh
    if np.linalg.det(random_rot) < 0:
        random_rot[2, :] *= -1
    geom = (random_rot @ geom.T).T
    xyz = "\n".join([f"H {x[0]} {x[1]} {x[2]}" for x in geom])

    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "d2h"


def test_pg_detection_c3v_2():
    xyz = """
    Fe 0 0 0
    Cl 0 0 1
    Cl 0 1 0
    Cl 1 0 0
    Br 0 0 -1
    Br 0 -1 0
    Br -1 0 0
    """
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "cs"


def test_pg_detection_d2d():
    xyz = """
    C 0 0 0
    H -1 1 0
    H -1 -1 0
    C 1 0 0
    H 2 0 1
    H 2 0 -1
    """

    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "c2v"


def test_pg_detection_c4v():
    xyz = """
    S 0 0 0
    F 1 0 0
    F 0 1 0
    F -1 0 0
    F 0 -1 0
    F 0 0 1
    Cl 0 0 -1"""
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "c2v"
