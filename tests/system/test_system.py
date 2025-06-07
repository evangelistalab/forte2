import pytest
import forte2
import numpy as np


def test_system():
    xyz = """

    H  0.0 0.0 0.0
    """

    system = forte2.System(xyz=xyz, basis="cc-pvdz")
    system_large_basis = forte2.System(xyz=xyz, basis="cc-pvdz")
    print(system)

    S = forte2.ints.overlap(system.basis)
    T = forte2.ints.kinetic(system.basis)
    V = forte2.ints.nuclear(system.basis, system.atoms)
    H = T + V

    # Solve the generalized eigenvalue problem H C = S C ε
    from scipy.linalg import eigh
    from numpy import isclose

    ε, _ = eigh(H, S)
    print("ε", ε)
    # assert isclose(ε[0], -0.4992784, atol=1e-7)

    M1 = forte2.ints.emultipole1(system.basis)
    print("S", S)
    print("M", M1)
    print(np.linalg.norm(S - M1[0]))

    M2 = forte2.ints.emultipole2(system.basis)
    # print("M2", M2)
    for i in range(4):
        print(np.linalg.norm(M1[i] - M2[i]))

    M3 = forte2.ints.emultipole3(system.basis)
    # print("M3", M3)
    for i in range(10):
        print(np.linalg.norm(M2[i] - M3[i]))

    opVop = forte2.ints.opVop(system.basis, system.atoms)
    print("opVop", opVop)


def test_xyz_comment():
    # Test an XYZ string with commented lines
    xyz = """
    #U 0.0 0.0 0.0
    @U 0.0 0.0 0.0
    H 0.0 0.0 0.0
    """

    # expect an exception to be raised
    system = forte2.System(xyz=xyz, basis="cc-pvdz")

    assert len(system.atoms) == 1


def test_missing_atom():
    # Test for missing atom in the basis set
    with pytest.raises(Exception) as excinfo:
        forte2.System(xyz="U 0.0 0.0 0.0", basis="cc-pvdz")
        assert (
            str(excinfo.value)
            == "[forte2] Basis set cc-pvdz does not contain element 92."
        )


def test_missing_coordinate():
    # Test for missing coordinates in the XYZ string
    with pytest.raises(ValueError) as excinfo:
        forte2.System(xyz="C 0.0 0.0", basis="cc-pvdz")

    with pytest.raises(ValueError) as excinfo:
        forte2.System(xyz="C 0.0", basis="cc-pvdz")


def test_units():
    # Test for different units
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """
    system = forte2.System(xyz=xyz, basis="cc-pvdz", unit="angstrom")
    assert system.unit == "angstrom"
    assert system.atoms[1][1][2] == pytest.approx(1.88972612)
    system = forte2.System(xyz=xyz, basis="cc-pvdz", unit="bohr")
    assert system.unit == "bohr"
    assert system.atoms[1][1][2] == pytest.approx(1.0)
