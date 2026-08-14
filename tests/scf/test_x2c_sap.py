import numpy as np
import pytest

from forte2 import System, X2CParams
from forte2.helpers.comparisons import approx
from forte2.scf import GHF, RHF


HBR = "H 0 0 0\nBr 0 0 1.4"


def _sap_system(x2c_type):
    return System(
        xyz=HBR,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        minao_basis_set=None,
        x2c=X2CParams(x2c_type=x2c_type, x2c_model="sap"),
    )


def test_invalid_x2c_option():
    with pytest.raises(ValueError, match="x2c must be an X2CParams instance"):
        System(xyz="H 0 0 0", basis_set="sto-3g", x2c="invalid")


def test_sfx2c_sap_hbr():
    system = _sap_system("sf")
    hcore = system.ints_hcore()
    assert np.allclose(hcore, hcore.T, atol=2.0e-12)

    scf = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system)
    scf.run()
    assert scf.E == approx(-2570.2025902236296)


def test_sox2c_sap_hbr():
    # The default SNSO option must not rescale SAP-X2C, whose model potential
    # already supplies both scalar and spin-orbit two-electron picture change.
    system = _sap_system("so")
    hcore = system.ints_hcore()
    assert np.allclose(hcore, hcore.conj().T, atol=3.0e-12)

    scf = GHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system)
    scf.run()
    assert scf.E == approx(-2570.172560851489)
