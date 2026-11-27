import json
from pathlib import Path

import pytest

from forte2 import AVAS, CISolver, MCOptimizer, RHF, State, System

CYCLOPROPENE_XYZ = """
H   0.912650   0.000000   1.457504
H  -0.912650   0.000000   1.457504
H   0.000000  -1.585659  -1.038624
H   0.000000   1.585659  -1.038624
C   0.000000   0.000000   0.859492
C   0.000000  -0.651229  -0.499559
C   0.000000   0.651229  -0.499559
"""


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).parent / "data" / "cyclopropene_sa2cas43_properties.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def cyclopropene():
    system = System(
        xyz=CYCLOPROPENE_XYZ,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
    )
    rhf = RHF(charge=0, e_tol=1e-10, d_tol=1e-8)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    ci = CISolver(
        State(nel=rhf.nel, multiplicity=1, ms=0.0), nroots=2, weights=[0.5, 0.5]
    )
    mc = MCOptimizer(
        ci,
        e_tol=1e-10,
        g_tol=1e-7,
        final_orbitals="original",
        do_transition_dipole=True,
    )(avas).run()
    return mc, rhf
