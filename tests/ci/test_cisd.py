import pytest
from forte2 import *

@pytest.mark.xfail(reason="CISD energy does not match RDM energy")
def test_cisd_1():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis="cc-pVDZ", auxiliary_basis="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CISD()(rhf)
    ci.run()
    assert ci.E[0] == pytest.approx(-100.2050066538116)
