import pytest
from forte2 import *
from forte2.helpers.comparisons import approx


@pytest.mark.slow
def test_cisd_1():
    escf = -99.9977252002953492
    ecisd = -100.204959657944

    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        auxiliary_basis_set_corr="cc-pVTZ-RIFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CISD()(rhf)
    ci.run()
    assert rhf.E == approx(escf)
    assert ci.E[0] == approx(ecisd)
