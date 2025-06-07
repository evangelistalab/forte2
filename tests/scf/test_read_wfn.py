import forte2
from forte2.scf import RHF
import copy
import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=1e-8, abs=5e-8)


def test_read_wfn():
    erhf = -76.061466407130
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system1 = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf1 = RHF(charge=0)(system1)
    scf1.run()
    assert scf1.E == approx(erhf)
    c0 = copy.deepcopy(scf1.C)

    e_newgeom = -76.061609428649
    xyz = """
    O            0.000000000000     0.000000000000    -0.063
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system2 = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    scf2 = RHF(charge=0)(system2)
    scf2.run()
    assert scf2.E == approx(e_newgeom)

    scf2.C = c0
    scf2.run()
    assert scf2.E == approx(e_newgeom)
