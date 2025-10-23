import numpy as np
import pytest

from forte2 import System, ints


def test_libcint_overlap():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = ints.cint_int1e_ovlp_sph(10, system.cint_atm, system.cint_bas, system.cint_env)
    assert np.linalg.norm(s) == pytest.approx(3.6556110774906956, rel=1e-6)


def test_libcint_kinetic():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = ints.cint_int1e_kin_sph(10, system.cint_atm, system.cint_bas, system.cint_env)
    assert np.linalg.norm(s) == pytest.approx(5.128923795496629, rel=1e-6)


def test_libcint_nuclear():
    xyz = """
    Ag 0 0 0
    Ag 0 0 1
    """
    system = System(
        xyz, basis_set="sto-3g", minao_basis_set=None, use_gaussian_charges=True
    )
    s = ints.cint_int1e_nuc_sph(
        system.nbf, system.cint_atm, system.cint_bas, system.cint_env
    )
    # from pyscf, using sto-3g downloaded from bse (the built-in one has different parameters..)
    assert np.linalg.norm(s) == pytest.approx(
        np.linalg.norm(3687.189758783181), rel=1e-6
    )
