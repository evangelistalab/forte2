import numpy as np
import pytest
from pathlib import Path

THIS_DIR = Path(__file__).parent


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
def test_libcint_overlap_cross():
    xyz = """
    Li 0 0 0
    Li 0 0 1.9
    """
    system = System(xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-jkfit")
    s_cint = integrals.cint_overlap(system, system.basis, system.auxiliary_basis)
    s_int2 = integrals.overlap(system, system.basis, system.auxiliary_basis)

    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(6.263577351975621, rel=1e-6)


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
    assert np.linalg.norm(s_cint[3] - c_int2[0]) < 1e-6  # sigma_x
    assert np.linalg.norm(s_cint[0] - c_int2[1]) < 1e-6  # sigma_y
    assert np.linalg.norm(s_cint[1] - c_int2[2]) < 1e-6  # sigma_z
    assert np.linalg.norm(s_cint[2] - c_int2[3]) < 1e-6  # I2
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
def test_libcint_2c2e_cross():
    xyz = """
    N 0 0 0
    N 0 0 1.1
    """
    system = System(xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    s_cint = integrals.cint_coulomb_2c(system, system.basis, system.auxiliary_basis)
    s_int2 = integrals.coulomb_2c(system, system.basis, system.auxiliary_basis)
    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(149.6527772441834, rel=1e-6)


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
def test_libcint_coulomb_3c():
    xyz = """
    O
    H 1 1.1
    H 1 1.1 2 104.5
    """
    system = System(xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    s_cint = integrals.cint_coulomb_3c(system)
    s_int2 = integrals.coulomb_3c(system)

    assert np.linalg.norm(s_cint - s_int2) < 1e-6
    assert np.linalg.norm(s_cint) == pytest.approx(60.4085268377979, rel=1e-6)


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_coulomb_3c_prealloc():
    xyz = """
    O
    H 1 1.1
    H 1 1.1 2 104.5
    """
    system = System(xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    ref = integrals.coulomb_3c(system)
    cint_computer = integrals.CInt3cBySlice(system)

    naux = len(system.auxiliary_basis)
    nb = len(system.basis)
    # random buffer that cannot hold the full integral tensor
    buf = np.zeros((naux - 10, nb - 5, nb - 5))
    nshaux = system.auxiliary_basis.nshells
    nshb = system.basis.nshells
    first_size_aux = system.auxiliary_basis.shell_first_and_size
    first_size_b = system.basis.shell_first_and_size

    rng = np.random.default_rng(12345)
    for _ in range(20):
        ish0 = rng.integers(0, nshaux - 1)
        ish1 = rng.integers(ish0 + 1, nshaux)
        jsh0 = rng.integers(0, nshb - 1)
        jsh1 = rng.integers(jsh0 + 1, nshb)
        ksh0 = rng.integers(0, nshb - 1)
        ksh1 = rng.integers(ksh0 + 1, nshb)

        ib0 = first_size_aux[ish0][0]
        ib1 = first_size_aux[ish1 - 1][0] + first_size_aux[ish1 - 1][1]
        jb0 = first_size_b[jsh0][0]
        jb1 = first_size_b[jsh1 - 1][0] + first_size_b[jsh1 - 1][1]
        kb0 = first_size_b[ksh0][0]
        kb1 = first_size_b[ksh1 - 1][0] + first_size_b[ksh1 - 1][1]

        if (
            ib1 - ib0 > buf.shape[0]
            or jb1 - jb0 > buf.shape[1]
            or kb1 - kb0 > buf.shape[2]
        ):
            continue

        shell_slices = [(ish0, ish1), (jsh0, jsh1), (ksh0, ksh1)]
        buf1 = cint_computer.compute(shell_slices, buf)
        assert np.linalg.norm(buf1 - ref[ib0:ib1, jb0:jb1, kb0:kb1]) < 1e-8


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_libcint_coulomb_3c_high_l():
    xyz = "H 0 0 0\nH 0 0 1.0"
    system = System(
        xyz,
        basis_set="sap_helfem_large",
        auxiliary_basis_set=str(THIS_DIR / "high_l.json"),
        minao_basis_set=None,
    )
    s_cint = integrals.coulomb_3c(system)
    assert np.linalg.norm(s_cint) == pytest.approx(15.857035710505492, rel=1e-6)
