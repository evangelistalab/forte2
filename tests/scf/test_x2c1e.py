import numpy as np
import pytest
import forte2
from forte2.scf import RHF, GHF
from forte2.helpers.comparisons import approx, approx_loose


def test_sfx2c1e():
    escf = -5192.021043979554
    xyz = """
    Br 0 0 0
    Br 0 0 1.2
    """

    system = forte2.System(
        xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT", x2c_type="sf"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)


def test_lindep_sfx2c1e():
    # psi4's x2c actually doesn't handle this case correctly
    # pyscf gives -4.071624245913899, so we need to investigate further
    # mol = pyscf.gto.M(
    #     atom=["H 0 0 %f" % i for i in range(10)],
    #     unit="Bohr",
    #     basis="aug-cc-pvdz",
    #     symmetry=False,
    # )
    # mf = pyscf.scf.RHF(mol).density_fit("cc-pvqz-jkfit").x2c()
    # mf = pyscf.scf.addons.remove_linear_dep_(mf, threshold=2e-7, lindep=1e-10)
    # mf.kernel()
    erhf = -4.071623764438

    xyz = "\n".join([f"H 0 0 {i}" for i in range(10)])

    system = forte2.System(
        xyz=xyz,
        basis="aug-cc-pvdz",
        auxiliary_basis="cc-pVQZ-JKFIT",
        unit="bohr",
        x2c_type="sf",
        ortho_thresh=2e-7,
    )

    scf = RHF(charge=0, econv=1e-10, dconv=1e-8)(system)
    scf.run()
    assert scf.E == approx(erhf)
    assert scf.nbf == 90
    assert scf.nmo == 81


def test_sox2c1e():
    eghf = -128.61570430672734
    xyz = "Ne 0 0 0"

    system = forte2.System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit", x2c_type="so"
    )
    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(eghf)
