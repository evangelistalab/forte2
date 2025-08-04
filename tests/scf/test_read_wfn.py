import copy

from forte2 import System
from forte2.scf import RHF
from forte2.helpers.comparisons import approx


def test_read_wfn():
    erhf = -76.061466407130
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system1 = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

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

    system2 = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf2 = RHF(charge=0)(system2)
    scf2.run()
    assert scf2.E == approx(e_newgeom)

    scf2.C = c0
    scf2.run()
    assert scf2.E == approx(e_newgeom)
