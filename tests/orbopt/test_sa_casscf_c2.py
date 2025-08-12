import pytest
import numpy as np

from forte2 import System, RHF, MCOptimizer, AVAS, State
from forte2.helpers.comparisons import approx


def test_sa_casscf_c2():
    """Test CASSCF with C2 molecule."""

    erhf = -75.382486499716
    ecasscf = -75.5580517997

    xyz = """
    C 0.0 0.0 0.0
    C 0.0 0.0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=4,
        num_active_uocc=4,
        subspace=["C(2s)", "C(2p)"],
    )(rhf)
    mc = MCOptimizer(State(nel=rhf.nel, multiplicity=1, ms=0.0), nroots=3)(avas)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(ecasscf)
