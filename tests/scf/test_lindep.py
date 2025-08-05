import numpy as np

from forte2 import System
from forte2.scf import RHF, GHF
from forte2.helpers.comparisons import approx


def test_lindep_rhf():
    erhf = -4.071545222979
    xyz = "\n".join([f"H 0 0 {i}" for i in range(10)])

    system = System(
        xyz=xyz,
        basis_set="aug-cc-pvdz",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        unit="bohr",
        ortho_thresh=2e-7,
    )

    ovlp = system.ints_overlap()
    assert np.linalg.cond(ovlp) > 1e14

    # test diis with linear dependency as well with tight convergence
    scf = RHF(charge=0, econv=1e-10, dconv=1e-8)(system)
    scf.run()
    assert scf.nbf == 90
    assert scf.nmo == 81
    assert scf.E == approx(erhf)


def test_lindep_ghf():
    erhf = -4.071545223158
    xyz = "\n".join([f"H 0 0 {i}" for i in range(10)])

    system = System(
        xyz=xyz,
        basis_set="aug-cc-pvdz",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        unit="bohr",
        ortho_thresh=2e-7,
    )

    ovlp = system.ints_overlap()
    assert np.linalg.cond(ovlp) > 1e14

    scf = GHF(charge=0)(system)

    scf.run()
    assert scf.nbf == 90
    assert scf.nmo == 81
    assert scf.E == approx(erhf)
