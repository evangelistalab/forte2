import numpy as np
import pytest

from forte2 import System, integrals
from forte2.system.build_basis import build_basis
from forte2.integrals import LIBCINT_AVAILABLE


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
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


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_ovlp_spinor():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s_cint = integrals.cint_overlap_spinor(system)
    assert np.linalg.norm(s_cint) == pytest.approx(5.169814764522727, rel=1e-6)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_spnucsp_sph_with_gaussian_charges():
    xyz = """
    Ag 0 0 0
    Ag 0 0 1
    """
    system = System(
        xyz, basis_set="sto-3g", use_gaussian_charges=True, minao_basis_set=None
    )
    s = integrals.cint_opVop(system)
    assert np.linalg.norm(s) == pytest.approx(5982385.012696481, rel=1e-6)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_spnucsp_sph():
    xyz = """
    Ag 0 0 0
    Ag 0 0 1
    """
    system = System(xyz, basis_set="sto-3g", minao_basis_set=None)
    s_cint = integrals.cint_opVop(system)
    c_int2 = integrals.opVop(system)
    assert np.linalg.norm(s_cint[3] - c_int2[0]) < 1e-6  # I2
    assert np.linalg.norm(s_cint[0] - c_int2[1]) < 1e-6  # sigma_x
    assert np.linalg.norm(s_cint[1] - c_int2[2]) < 1e-6  # sigma_y
    assert np.linalg.norm(s_cint[2] - c_int2[3]) < 1e-6  # sigma_z
    assert np.linalg.norm(s_cint) == pytest.approx(5982385.234519612, rel=1e-6)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_spnucsp_spinor():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    s = integrals.cint_opVop_spinor(system)
    assert np.linalg.norm(s) == pytest.approx(116.46738183606718, rel=1e-6)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
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


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_kinetic_decontracted():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set={"Li1": "decon-sto-3g", "Li2": "cc-pvdz"})
    xbasis = build_basis("sto-3g", system.geom_helper, decontract=True)
    s_cint = integrals.cint_kinetic(system, basis1=xbasis)
    s_int2 = integrals.kinetic(system, basis1=xbasis)

    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(36.523146675022836, rel=1e-6)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
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


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
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


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
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


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_r_sph_shifted_origin():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g")
    origin = [0.1, -0.2, 0.3]
    s_cint = integrals.cint_emultipole1(system, origin=origin)
    s_int2 = integrals.emultipole1(system, origin=origin)
    for i in range(3):
        # s_int2 = [overlap, x, y, z], so skip the zeroth element
        assert np.linalg.norm(s_cint[i] - s_int2[i + 1]) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(11.116795764727945, rel=1e-6)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_sprsp_sph_with_shifted_origin():
    xyz = """
    H 0 0 0
    Br 0 0 1.2
    """
    system = System(xyz, basis_set="ano-rcc", minao_basis_set=None)
    s_cint = integrals.cint_sprsp(system, origin=[0.1, -0.2, 0.3])
    # mol = pyscf.M(atom="H 0 0 0; Br 0 0 1.2", basis="ano-rcc.nw", spin=0, charge=0)
    # with mol.with_common_orig((0.1, -0.2, 0.3)):
    #     integrals = mol.intor("int1e_sprsp_sph")
    #     for i in integrals:
    #         print(np.linalg.norm(i))
    norm_ref = [
        0.0,
        26.026312334673438,
        25.724080350429066,
        205.49749994791185,
        26.026312334673438,
        0.0,
        25.724080350429066,
        386.2822061421359,
        25.724080350429066,
        25.724080350429062,
        0.0,
        3713.7650944042684,
    ]
    for i in range(12):
        assert np.linalg.norm(s_cint[i]) == pytest.approx(norm_ref[i], rel=1e-6)
