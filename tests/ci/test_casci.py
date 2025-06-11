import pytest
from forte2 import *


# equivalence to test_ci_rhf.py::test_ci_2
def test_casci_1():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CASCI(6, 8)(rhf)
    ci.run()

    assert ci.E[0] == pytest.approx(-100.019788438077)
