import json
import re
from importlib import resources

import pytest

from forte2 import System

# Symmetric molecules from the Otterbein gallery, randomly rotated; see README.md.
with (
    resources.files("forte2.data").joinpath("otterbein_symmetry_db.json").open("r") as f
):
    OTTERBEIN_DB = json.load(f)

OTTERBEIN_SLOW = {
    "20",
    "27",
    "30",
    "81",
    "93",
    "94",
    "105",
    "110",
    "111",
    "112",
    "118",
    "126",
    "128",
}
OTTERBEIN_SYMMETRY_TOL = {"31": 1e-5}


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


def _otterbein_param(key):
    marks = [pytest.mark.slow] if key in OTTERBEIN_SLOW else []
    return pytest.param(key, marks=marks, id=key)


@pytest.mark.parametrize("key", [_otterbein_param(key) for key in OTTERBEIN_DB])
def test_pg_detection_otterbein(key):
    mol = OTTERBEIN_DB[key]
    # ANO-R0 has no Th basis.
    basis = {"default": "ano-r0"}
    if re.search(r"\bTh\b", mol["xyz"]):
        basis["Th"] = "sap_helfem_large"
    system = System(
        xyz=mol["xyz"],
        basis_set=basis,
        minao_basis_set=None,
        symmetry=True,
        symmetry_tol=OTTERBEIN_SYMMETRY_TOL.get(key, 1e-4),
    )
    assert system.point_group.lower() == mol["abelian_pg"].lower()
    # Raises unless the group's operations hold in the standard orientation.
    _ = system.symmetry_basis
