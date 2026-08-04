import numpy as np
import pytest

from forte2 import System
from forte2.scf import RHF, GHF, UHF
from forte2.helpers.comparisons import approx
from forte2.orbitals import convert_coeff_spatial_to_spinor
from forte2.system import BSE_AVAILABLE
from forte2.data import EH_TO_WN, EH_TO_EV


def _four_point_hcore_deriv(system_factory, coordinate, step=1.0e-4):
    values = [
        system_factory(coordinate + scale * step).ints_hcore()
        for scale in (-2.0, -1.0, 1.0, 2.0)
    ]
    return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * step)


@pytest.mark.parametrize("x2c_type", ["sf", "so"])
@pytest.mark.parametrize("use_gaussian_charges", [False, True])
def test_x2c_hcore_deriv_finite_difference(x2c_type, use_gaussian_charges):
    def make_system(z):
        return System(
            xyz=f"O 0 0 0\nH 0 0 {z:.12f}\nH 0 1.4 0",
            basis_set="sto-3g",
            unit="bohr",
            x2c_type=x2c_type,
            use_gaussian_charges=use_gaussian_charges,
            minao_basis_set=None,
        )

    system = make_system(1.5)
    analytical = system.x2c_helper.hcore_deriv()[5]
    numerical = _four_point_hcore_deriv(make_system, 1.5)

    assert analytical == pytest.approx(numerical, abs=3.0e-8)
    assert analytical == pytest.approx(analytical.conj().T, abs=1.0e-12)


@pytest.mark.parametrize("snso_type", ["boettger", "dc", "dcb", "row-dependent"])
def test_snso_x2c_hcore_deriv_finite_difference(snso_type):
    def make_system(z):
        return System(
            xyz=f"S 0 0 0\nH 0 0 {z:.12f}\nH 0 1.4 0",
            basis_set="sto-3g",
            unit="bohr",
            x2c_type="so",
            snso_type=snso_type,
            minao_basis_set=None,
        )

    system = make_system(1.5)
    analytical = system.x2c_helper.hcore_deriv()[5]
    numerical = _four_point_hcore_deriv(make_system, 1.5)

    assert analytical == pytest.approx(numerical, abs=3.0e-8)


def test_x2c_hcore_deriv_with_truncated_overlap_space():
    def make_system(z):
        return System(
            xyz=f"O 0 0 0\nH 0 0 {z:.12f}\nH 0 1.4 0",
            basis_set="sto-3g",
            unit="bohr",
            x2c_type="sf",
            overlap_ortho_rtol=1.0e-3,
            minao_basis_set=None,
        )

    system = make_system(1.5)
    assert system.x2c_helper.orth_info["n_discarded"] == 1
    analytical = system.x2c_helper.hcore_deriv()[5]
    numerical = _four_point_hcore_deriv(make_system, 1.5)

    assert analytical == pytest.approx(numerical, abs=3.0e-8)


def test_x2c_helper_tracks_spinor_upcaster_override():
    def make_system(x2c_type, snso_type=None):
        return System(
            xyz="S 0 0 0\nH 0 0 1.5",
            basis_set="sto-3g",
            unit="bohr",
            x2c_type=x2c_type,
            snso_type=snso_type,
            minao_basis_set=None,
        )

    overridden = make_system("sf")
    overridden.x2c_type = "so"
    overridden.snso_type = "row-dependent"
    reference = make_system("so", "row-dependent")

    assert overridden.x2c_helper.hcore_x2c() == pytest.approx(
        reference.x2c_helper.hcore_x2c(), abs=1.0e-12
    )
    assert overridden.x2c_helper.hcore_deriv() == pytest.approx(
        reference.x2c_helper.hcore_deriv(), abs=1.0e-11
    )


def test_sfx2c1e():
    escf = -5192.021043979554
    xyz = """
    Br 0 0 0
    Br 0 0 1.2
    """

    system = System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT", x2c_type="sf"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)


def test_sfx2c1e_with_gaussian_charges():
    escf = -5192.003129895058
    xyz = """
    Br 0 0 0
    Br 0 0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c_type="sf",
        use_gaussian_charges=True,
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(escf)


@pytest.mark.skip(reason="This test cannot be reliably reproduced.")
def test_lindep_sfx2c1e():
    # psi4's x2c actually doesn't handle this case correctly
    # pyscf gives -4.071624245913899, so we need to investigate further
    # mol = pyscf.gto.M(
    #     atom=["H 0 0 %f" % i for i in range(10)],
    #     unit="Bohr",
    #     basis_set="aug-cc-pvdz",
    #     symmetry=False,
    # )
    # mf = pyscf.scf.RHF(mol).density_fit("cc-pvqz-jkfit").x2c()
    # mf = pyscf.scf.addons.remove_linear_dep_(mf, threshold=2e-7, lindep=1e-10)
    # mf.kernel()
    erhf = -4.071623764438

    xyz = "\n".join([f"H 0 0 {i}" for i in range(10)])

    system = System(
        xyz=xyz,
        basis_set="aug-cc-pvdz",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        unit="bohr",
        x2c_type="sf",
        overlap_ortho_rtol=2e-7,
    )

    scf = RHF(charge=0, e_tol=1e-10, d_tol=1e-8)(system)
    scf.run()
    assert scf.E == approx(erhf)
    assert scf.nbf == 90
    assert scf.nmo == 81


def test_sox2c1e_water():
    eghf = -76.08194686989626
    xyz = """
    O 0 0 0
    H 0 -0.757 0.587
    H 0 0.757 0.587
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c_type="so",
        snso_type=None,
    )
    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(eghf)


def test_boettger_hbr():
    xyz = """
    H 0 0 0
    Br 0 0 1.4
    """

    system = System(
        xyz=xyz,
        basis_set={"Br": "decon-aug-cc-pvdz", "default": "cc-pvtz"},
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c_type="so",
        snso_type="dcb",
    )
    scf = GHF(charge=0)(system)
    scf.run()
    assert EH_TO_WN * (
        scf.eps[0][scf.nel - 2] - scf.eps[0][scf.nel - 3]
    ) == pytest.approx(2953.193840819996, abs=1e-4)


def test_so_from_sf_water():
    euhf = -75.711680104122
    eghf = -75.7116858952105
    xyz = """
    O 0 0 0
    H 0 -0.757 0.587
    H 0 0.757 0.587
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c_type="sf",
    )
    scf = UHF(charge=1, ms=0.5)(system)
    scf.run()
    assert scf.E == approx(euhf)

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c_type="so",
        snso_type=None,
    )
    scf_so = GHF(charge=1)(system)
    scf_so.C = convert_coeff_spatial_to_spinor(scf.C)
    scf_so.run()
    assert scf_so.E == approx(eghf)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="Basis set exchange is not available")
def test_sox2c1e_sc():
    l23_ref = 4.395077289942328
    xyz = "Sc 0 0 0"
    system = System(
        xyz=xyz,
        basis_set="sapporo-dkh3-dzp-2012-diffuse",
        auxiliary_basis_set="def2-universal-jkfit",
        x2c_type="so",
        snso_type="row-dependent",
    )
    scf = GHF(charge=3)(system)
    scf.run()
    l23_splitting = EH_TO_EV * (scf.eps[0][6] - scf.eps[0][5])
    assert l23_splitting == pytest.approx(l23_ref, abs=1e-5)
