import numpy as np
import forte2
from forte2.scf import RHF, GHF
from forte2.helpers.comparisons import approx


def test_lindep_rhf():
    erhf = -4.071545223158
    xyz = "\n".join([f"H 0 0 {i}" for i in range(10)])

    system = forte2.System(
        xyz=xyz, basis="aug-cc-pvdz", auxiliary_basis="cc-pVQZ-JKFIT", unit="bohr"
    )

    ovlp = system.ints_overlap()
    assert np.linalg.cond(ovlp) > 1e14

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.nbf == 90
    assert scf.nmo == 81
    assert scf.E == approx(erhf)


def test_lindep_ghf():
    erhf = -4.071545223158
    xyz = "\n".join([f"H 0 0 {i}" for i in range(10)])

    system = forte2.System(
        xyz=xyz, basis="aug-cc-pvdz", auxiliary_basis="cc-pVQZ-JKFIT", unit="bohr"
    )

    ovlp = system.ints_overlap()
    assert np.linalg.cond(ovlp) > 1e14

    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.nbf == 90
    assert scf.nmo == 81
    assert scf.E == approx(erhf)
