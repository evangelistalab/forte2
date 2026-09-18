import importlib
import inspect
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
import pytest

from forte2 import RHF, System, X2CParams
from forte2.lib import ints
from forte2.integrals import LIBCINT_AVAILABLE
from forte2.system import coords_to_atoms, coords_to_xyz

INTEGRALS_TEST_DIR = Path(__file__).parent.parent / "integrals"


def test_x2c_is_the_only_public_relativistic_option():
    parameters = inspect.signature(System).parameters
    assert "x2c" in parameters
    assert "x2c_type" not in parameters
    assert "x2c_model" not in parameters
    assert "snso_type" not in parameters


@pytest.mark.parametrize("option", ["so-1e", "invalid", 1])
def test_invalid_x2c_option(option):
    with pytest.raises(ValueError, match="x2c must be an X2CParams instance"):
        System(xyz="H 0 0 0", basis_set="sto-3g", minao_basis_set=None, x2c=option)


def test_save_load_roundtrips_x2c_params(tmp_path):
    filename = tmp_path / "x2c_system"
    system = System(
        xyz="H 0 0 0",
        basis_set="sto-3g",
        minao_basis_set=None,
        x2c=X2CParams(x2c_type="so", x2c_model="1e", snso_type="dcb"),
    )
    system.save(filename)

    loaded = System.load(filename)
    assert isinstance(loaded.x2c, X2CParams)
    assert loaded.x2c_type == "so"
    assert loaded.x2c_model == "1e"
    assert loaded.snso_type == "dcb"


def test_system():
    xyz = """

    H  0.0 0.0 0.0
    """

    system = System(xyz=xyz, basis_set="cc-pvdz")
    print(system)

    S = ints.overlap(system.basis)
    T = ints.kinetic(system.basis)
    V = ints.nuclear(system.basis, system.atoms)
    H = T + V

    # Solve the generalized eigenvalue problem H C = S C ε
    ε, _ = eigh(H, S)
    print("ε", ε)
    # assert isclose(ε[0], -0.4992784, atol=1e-7)

    M1 = ints.emultipole1(system.basis)
    print("S", S)
    print("M", M1)
    print(np.linalg.norm(S - M1[0]))

    M2 = ints.emultipole2(system.basis)
    # print("M2", M2)
    for i in range(4):
        print(np.linalg.norm(M1[i] - M2[i]))

    M3 = ints.emultipole3(system.basis)
    # print("M3", M3)
    for i in range(10):
        print(np.linalg.norm(M2[i] - M3[i]))

    opVop = ints.opVop(system.basis, system.atoms)
    print("opVop", opVop)


@pytest.mark.parametrize("atom", ["Na", "NA", "na", "nA"])
def test_xyz_with_caps(atom):
    # Test an XYZ string with capitalized atom names
    xyz = f"""
    {atom} 0.0 0.0 0.0
    {atom} 0.0 0.0 2.5
    """
    system = System(xyz=xyz, basis_set="cc-pvdz")
    assert len(system.atoms) == 2


def test_xyz_comment():
    # Test an XYZ string with commented lines
    xyz = """
    #U 0.0 0.0 0.0
    @U 0.0 0.0 0.0
    H 0.0 0.0 0.0
    """
    system = System(xyz=xyz, basis_set="cc-pvdz")

    assert len(system.atoms) == 1


def test_missing_atom():
    # Test for missing atom in the basis set
    with pytest.raises(Exception) as excinfo:
        System(xyz="U 0.0 0.0 0.0", basis_set="cc-pvdz")
        assert (
            str(excinfo.value)
            == "[forte2] Basis set cc-pvdz does not contain element 92."
        )


def test_missing_coordinate():
    # Test for missing coordinates in the XYZ string
    with pytest.raises(ValueError) as excinfo:
        System(xyz="C 0.0 0.0", basis_set="cc-pvdz")

    with pytest.raises(ValueError) as excinfo:
        System(xyz="C 0.0", basis_set="cc-pvdz")


def test_units():
    # Test for different units
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", unit="angstrom")
    assert system.unit == "angstrom"
    assert system.atoms[1][1][2] == pytest.approx(1.88972612)
    system = System(xyz=xyz, basis_set="cc-pvdz", unit="bohr")
    assert system.unit == "bohr"
    assert system.atoms[1][1][2] == pytest.approx(1.0)


def test_nuclear_dipole():
    # Test for nuclear dipole calculation
    xyz = """
    O 0.0 0.0 0.0
    H 0.0 0.0 1.0
    Li 2.0 0.0 0.0
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", unit="bohr")

    nucdip = system.nuclear_dipole(unit="au")
    assert nucdip == pytest.approx([6.0, 0.0, 1.0])


def test_nuclear_quadrupole():
    # Test for nuclear quadrupole calculation
    xyz = """
    O 0.0 0.0 0.0
    H 0.0 0.0 1.0
    Li 2.0 0.0 0.0
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", unit="bohr")

    nucquad = system.nuclear_quadrupole(unit="au")
    assert np.trace(nucquad) == pytest.approx(0.0)
    assert np.diag(nucquad) == pytest.approx([11.5, -6.5, -5.0])


def test_nuclear_moments_do_not_mutate_geometry():
    xyz = """
    O 0.0 0.0 0.0
    H 0.0 0.0 1.0
    Li 2.0 0.0 0.0
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", unit="bohr")
    positions_before = np.array(system.atomic_positions, copy=True)

    origin = (0.5, -1.0, 2.0)
    system.nuclear_dipole(origin=origin, unit="au")
    assert np.allclose(system.atomic_positions, positions_before)

    system.nuclear_quadrupole(origin=origin, unit="au")
    assert np.allclose(system.atomic_positions, positions_before)

    # And the origin shift must still be applied to the *returned* result:
    dip_shifted = system.nuclear_dipole(origin=origin, unit="au")
    dip_ref = system.nuclear_dipole(unit="au") - np.array(origin) * system.Zsum
    assert dip_shifted == pytest.approx(dip_ref)


def test_center_of_mass():
    # Test for center of mass calculation
    xyz = """
    O 0.0 0.0 0.0
    H 0.0 0.0 1.0
    Li 2.0 0.0 0.0
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", unit="bohr")
    com = system.center_of_mass
    assert com == pytest.approx([0.57948887, 0, 0.04208937])

    # This also incidentally tests the input of geometry with signed integers
    xyz = """
    H  1  1  1
    P  1  1 -1
    V  1 -1  1
    Mn  1 -1 -1
    Mn -1  1  1
    V -1  1 -1
    P -1 -1  1
    H -1 -1 -1 
    """
    system = System(xyz=xyz, basis_set="cc-pvdz", unit="bohr")
    com = system.center_of_mass
    assert com == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)


def test_custom_basis_rhf():
    xyz = """
    C 0 0 0
    O 0 0 1.2
    """
    system = System(
        xyz=xyz,
        basis_set={"C": "cc-pvdz", "O": "sto-6g"},
        auxiliary_basis_set={"C": "cc-pVQZ-JKFIT", "O": "def2-universal-JKFIT"},
    )
    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == pytest.approx(-112.484140615262, rel=1e-8, abs=1e-8)


def test_missing_auxiliary_basis():
    xyz = """
    C 0 0 0
    O 0 0 1.2
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
    )
    scf = RHF(charge=0)(system)
    # verify that an error is raised when the auxiliary basis is missing
    with pytest.raises(ValueError, match="Auxiliary basis is not defined"):
        scf.run()


def test_zmatrix_0():
    # Test for Z-matrix input, with mixed line breaks, spacings and indentations
    zmat = """
    C;C    1    1.333
        H    1    1.079    2    121.4
H 1    1.079    2 121.4    3    180.0
    H    2    1.079    1    121.4    3      0.0
            H    2    1.079    1    121.4    3    180.0
    """
    system = System(
        xyz=zmat,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="angstrom",
    )
    scf = RHF(charge=0)(system)
    scf.run()

    def z_to_cart_0():
        """
        Psi4 geometry
        """
        xyz = """
            C            0.000000000000     0.000000000000    -0.666500000000
            C            0.000000000000     0.000000000000     0.666500000000
            H            0.000000000000     0.920981310260    -1.228669392756
            H            0.000000000000    -0.920981310260    -1.228669392756
            H            0.000000000000     0.920981310260     1.228669392756
            H            0.000000000000    -0.920981310260     1.228669392756
        """
        system = System(
            xyz=xyz,
            basis_set="cc-pvdz",
            auxiliary_basis_set="def2-universal-JKFIT",
            unit="angstrom",
        )
        scf = RHF(charge=0)(system)
        scf.run()
        return scf.E

    E_ref = z_to_cart_0()

    assert scf.E == pytest.approx(E_ref, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("atom", ["Na", "NA", "na", "nA"])
def test_zmatrix_with_caps(atom):
    # Test an zmatrix string with capitalized atom names
    xyz = f"""
    {atom}
    {atom} 1 1.2
    """
    system = System(xyz=xyz, basis_set="cc-pvdz")
    assert len(system.atoms) == 2


def test_ghost_atom():
    xyz = """
    H 0 0 0
    H 0 0 1.0
    X 0 0 0.5
    """
    system = System(
        xyz=xyz,
        basis_set={"H": "sto-3g", "X": "decon-cc-pvqz::H"},
        auxiliary_basis_set={"default": "def2-universal-JKFIT::H"},
        minao_basis_set={"default": "cc-pvtz-minao::H"},
    )
    assert len(system.basis) == 34
    scf = RHF(charge=0, guess_type="hcore")(system)
    scf.run()
    assert scf.E == pytest.approx(-1.096696530945, rel=1e-8, abs=1e-8)


def test_backend_option_default_is_auto():
    system = System(xyz="H 0 0 0", basis_set="sto-3g", minao_basis_set=None)
    assert system.integral_backend == "auto"


def test_invalid_backend_option():
    with pytest.raises(ValueError, match="Invalid integral_backend"):
        System(
            xyz="H 0 0 0",
            basis_set="sto-3g",
            minao_basis_set=None,
            integral_backend="not-a-backend",
        )


def test_backend_libint2_rejects_high_l_basis():
    """Forcing the libint2 backend on a basis it cannot handle raises immediately,
    during System construction (the overlap matrix is built in __post_init__)."""
    with pytest.raises(ValueError, match="libint2 backend cannot handle"):
        System(
            xyz="H 0 0 0\nH 0 0 1.0",
            basis_set=str(INTEGRALS_TEST_DIR / "high_l.json"),
            minao_basis_set=None,
            integral_backend="libint2",
        )


@pytest.mark.skipif(not LIBCINT_AVAILABLE, reason="Libcint is not available")
def test_backend_libcint_used_even_for_low_l_basis(monkeypatch):
    """Explicitly requesting the libcint backend must route even ordinary
    low-angular-momentum integrals through Libcint, not just the high-l fallback."""
    integrals_impl = importlib.import_module("forte2.integrals.integrals")
    calls = []
    original_cint_overlap = integrals_impl.cint_overlap

    def traced_cint_overlap(*args, **kwargs):
        calls.append(1)
        return original_cint_overlap(*args, **kwargs)

    monkeypatch.setattr(integrals_impl, "cint_overlap", traced_cint_overlap)

    system = System(
        xyz="H 0 0 0\nH 0 0 1.0",
        basis_set="sto-3g",
        minao_basis_set=None,
        integral_backend="libcint",
    )
    assert len(calls) >= 1
    assert system.basis.max_l == 0  # sto-3g on H is well within libint2's range


def test_backend_option_save_load_roundtrip(tmp_path):
    filename = tmp_path / "backend_system"
    system = System(
        xyz="H 0 0 0",
        basis_set="sto-3g",
        minao_basis_set=None,
        integral_backend="libint2",
    )
    system.save(filename)

    loaded = System.load(filename)
    assert loaded.integral_backend == "libint2"


### begin test setup for with_geometry
_H2O_CHARGES = [8, 1, 1]
_H2O_COORDINATES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.6, 0.0, 0.0]])


def _h2o(coordinates=None, **kwargs):
    kwargs.setdefault("basis_set", "sto-3g")
    kwargs.setdefault("auxiliary_basis_set", "def2-universal-JKFIT")
    kwargs.setdefault("unit", "bohr")
    if coordinates is None:
        coordinates = _H2O_COORDINATES
    return System(xyz=coords_to_xyz(_H2O_CHARGES, coordinates), **kwargs)


def _displaced(system, atom=1, cart=2, step=1.0e-3):
    coordinates = system.atomic_positions.copy()
    coordinates[atom, cart] += step
    return coordinates


### end test setup for with_geometry


def test_coords_to_xyz_round_trips_through_parse():
    charges = [8, 1, 1]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [1.6, 0.0, 0.0]])
    system = System(
        xyz=coords_to_xyz(charges, coordinates),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    assert np.allclose(system.atomic_positions, coordinates)
    assert np.array_equal(system.atomic_charges, charges)


def test_coords_to_atoms_converts_angstrom_and_matches_parse():
    charges = [1, 1]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    atoms = coords_to_atoms(charges, coordinates, unit="angstrom")
    reference = System(
        xyz=coords_to_xyz(charges, coordinates),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="angstrom",
    ).atoms

    assert [atom[0] for atom in atoms] == [atom[0] for atom in reference]
    assert np.allclose(
        [atom[1] for atom in atoms], [atom[1] for atom in reference], atol=1.0e-14
    )


def test_with_geometry():
    system = _h2o()
    coordinates = _displaced(system)

    moved = system.with_geometry(coordinates)

    assert np.array_equal(system.atomic_positions, _H2O_COORDINATES)
    assert np.array_equal(moved.atomic_positions, coordinates)
    assert np.array_equal(moved.atomic_charges, system.atomic_charges)


def test_with_geometry_is_chainable():
    system = _h2o()
    step = np.zeros_like(system.atomic_positions)
    step[2, 0] = 0.01

    once = system.with_geometry(system.atomic_positions + step)
    twice = once.with_geometry(once.atomic_positions + step)
    direct = system.with_geometry(system.atomic_positions + 2 * step)

    assert np.allclose(twice.atomic_positions, direct.atomic_positions, atol=1.0e-14)
    assert twice.nuclear_repulsion == pytest.approx(
        direct.nuclear_repulsion, abs=1.0e-14
    )
