from forte2 import System


def test_pg_detection_atom():
    xyz = """
    H 0 0 0
    """
    system = System(xyz=xyz, basis_set="sto-6g", symmetry=True)
    assert system.point_group.lower() == "d2h"
