import numpy as np
import pytest

from forte2 import System, integrals


def test_libcint_overlap():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_overlap(system)
    assert np.linalg.norm(s) == pytest.approx(3.6556110774906956, rel=1e-6)


def test_libcint_ovlp_spinor():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_overlap_spinor(system)
    assert np.linalg.norm(s) == pytest.approx(5.169814764522727, rel=1e-6)


def test_libcint_spnucsp_spinor():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_opVop_spinor(system)
    assert np.linalg.norm(s) == pytest.approx(116.46738183606718, rel=1e-6)


def test_libcint_kinetic():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_kinetic(system)
    assert np.linalg.norm(s) == pytest.approx(5.128923795496629, rel=1e-6)


def test_libcint_nuclear():
    xyz = """
    Ag 0 0 0
    Ag 0 0 1
    """
    system = System(
        xyz, basis_set="sto-3g", minao_basis_set=None, use_gaussian_charges=True
    )
    s = integrals.cint_nuclear(system)
    # from pyscf, using sto-3g downloaded from bse (the built-in one has different parameters..)
    assert np.linalg.norm(s) == pytest.approx(
        np.linalg.norm(3687.189758783181), rel=1e-6
    )


def test_libcint_2c2e():
    xyz = """
    N 0 0 0
    N 0 0 1.1
    """
    system = System(xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    s = integrals.cint_coulomb_2c(system)
    assert np.linalg.norm(s) == pytest.approx(159.31789654133004, rel=1e-6)


def test_libcint_r_sph():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_emultipole1(system)
    assert np.linalg.norm(s) == pytest.approx(11.66647604433945, rel=1e-6)
