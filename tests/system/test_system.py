import inspect
import json

import numpy as np
from scipy.linalg import eigh
import pytest

from forte2 import System, RHF
from forte2.lib import ints
from forte2.system.system import parse_x2c_option


@pytest.mark.parametrize(
    "option, parsed",
    [
        (None, (None, None, None, None)),
        ("sf-1e", ("sf-1e", "sf", "1e", None)),
        ("so-1e", ("so-1e", "so", "1e", None)),
        (
            "so-snso-boettger",
            ("so-snso-boettger", "so", "1e", "boettger"),
        ),
        ("so-snso-dc", ("so-snso-dc", "so", "1e", "dc")),
        ("so-snso-dcb", ("so-snso-dcb", "so", "1e", "dcb")),
        (
            "so-snso-row-dependent",
            ("so-snso-row-dependent", "so", "1e", "row-dependent"),
        ),
        ("sf-sap", ("sf-sap", "sf", "sap", None)),
        ("so-sap", ("so-sap", "so", "sap", None)),
    ],
)
def test_x2c_option_parsing(option, parsed):
    assert parse_x2c_option(option) == parsed


def test_x2c_option_is_case_insensitive():
    assert parse_x2c_option("SO-SAP") == ("so-sap", "so", "sap", None)


def test_x2c_is_the_only_public_relativistic_option():
    parameters = inspect.signature(System).parameters
    assert "x2c" in parameters
    assert "x2c_type" not in parameters
    assert "x2c_model" not in parameters
    assert "snso_type" not in parameters


@pytest.mark.parametrize("option", ["invalid", "sf-sap-extra", 1])
def test_invalid_x2c_option(option):
    with pytest.raises(ValueError, match="x2c"):
        parse_x2c_option(option)


def test_load_migrates_legacy_x2c_options(tmp_path):
    filename = tmp_path / "legacy_x2c_system"
    system = System(
        xyz="H 0 0 0",
        basis_set="sto-3g",
        minao_basis_set=None,
        x2c="so-snso-dcb",
    )
    system.save(filename)

    json_file = filename.with_suffix(".json")
    data = json.loads(json_file.read_text(encoding="utf-8"))
    init_args = data["init_args"]
    init_args.pop("x2c")
    init_args.update(x2c_type="so", x2c_model="1e", snso_type="dcb")
    json_file.write_text(json.dumps(data), encoding="utf-8")

    loaded = System.load(filename)
    assert loaded.x2c == "so-snso-dcb"
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


def test_xyz_comment():
    # Test an XYZ string with commented lines
    xyz = """
    #U 0.0 0.0 0.0
    @U 0.0 0.0 0.0
    H 0.0 0.0 0.0
    """

    # expect an exception to be raised
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
