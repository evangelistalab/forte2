import numpy as np
import pytest

from forte2 import System, integrals


def test_libcint_overlap():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s_cint = integrals.cint_overlap(system)
    s_int2 = integrals.overlap(system)
    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(3.6556110774906956, rel=1e-6)


def test_libcint_ovlp_spinor():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s_cint = integrals.cint_overlap_spinor(system)
    assert np.linalg.norm(s_cint) == pytest.approx(5.169814764522727, rel=1e-6)


def test_libcint_spnucsp_sph():
    xyz = """
    Ag 0 0 0
    Ag 0 0 1
    """
    system = System(
        xyz, basis_set="sto-3g", use_gaussian_charges=True, minao_basis_set=None
    )
    s = integrals.cint_opVop(system)
    assert np.linalg.norm(s) == pytest.approx(5982385.012696481, rel=1e-6)


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
    s_cint = integrals.cint_kinetic(system)
    s_int2 = integrals.kinetic(system)
    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(5.128923795496629, rel=1e-6)


def test_libcint_nuclear():
    xyz = """
    Ag 0 0 0
    Ag 0 0 1
    """
    system = System(
        xyz, basis_set="sto-3g", minao_basis_set=None, use_gaussian_charges=True
    )
    s_cint = integrals.cint_nuclear(system)
    s_int2 = integrals.nuclear(system)
    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    # from pyscf, using sto-3g downloaded from bse (the built-in one has different parameters..)
    assert np.linalg.norm(s_cint) == pytest.approx(
        np.linalg.norm(3687.189758783181), rel=1e-6
    )


def test_libcint_2c2e():
    xyz = """
    N 0 0 0
    N 0 0 1.1
    """
    system = System(xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    s_cint = integrals.cint_coulomb_2c(system)
    s_int2 = integrals.coulomb_2c(system)
    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(159.31789654133004, rel=1e-6)


def test_libcint_r_sph():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s_cint = integrals.cint_emultipole1(system)
    s_int2 = integrals.emultipole1(system)
    for i in range(3):
        # s_int2 = [overlap, x, y, z], so skip the zeroth element
        assert np.linalg.norm(s_cint[i] - s_int2[i + 1]) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(11.66647604433945, rel=1e-6)
