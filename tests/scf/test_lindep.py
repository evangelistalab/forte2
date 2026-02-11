import numpy as np

from forte2 import System
from forte2.scf import RHF, GHF
from forte2.helpers.comparisons import approx, approx_loose


def test_lindep_rhf():
    erhf = -4.071076569404
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
    assert scf.nmo == 79
    assert scf.E == approx(erhf)


def test_lindep_ghf():
    erhf = -4.071076569404
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
    assert scf.nmo == 79
    assert scf.E == approx(erhf)


def test_lindep_x2c():
    # This tests the handling of linear dependencies in the X2C transformation
    # The basis sets are decontracted during X2C, resulting in cond(S) ~ 8e9.
    eref = -20264.784349176811
    xyz = """
    Tl 0 0 1.4
    H 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set={"H": "aug-cc-pVTZ", "Tl": "x2c-tzvpall-2c"},
        auxiliary_basis_set={
            "H": "aug-cc-pVTZ-autoaux",
            "Tl": "x2c-tzvpall-2c-autoaux",
        },
        minao_basis_set="ano-r0",
        x2c_type="so",
        snso_type="row-dependent",
        use_gaussian_charges=True,
        ortho_thresh=5e-10,
    )
    scf = GHF(charge=0, econv=1e-10, dconv=1e-8)(system)
    scf.run()
    assert scf.E == approx_loose(eref)
