import pytest
import forte2


def test_basis_center_first_and_last():
    # Test the Basis class
    basis = forte2.ints.Basis()
    # center 1: 1 + 3 = 4 basis functions, range (0, 4)
    basis.add(forte2.ints.Shell(0, [1.0], [1.0], [0.0, 0.0, 0.0]))
    basis.add(forte2.ints.Shell(1, [1.0], [1.0], [0.0, 0.0, 0.0]))
    # center 2: 1 + 5 = 6 basis functions, range (4, 10)
    basis.add(forte2.ints.Shell(0, [1.0], [1.0], [0.0, 0.0, 1.0]))
    basis.add(forte2.ints.Shell(2, [1.0], [1.0], [0.0, 0.0, 1.0]))
    # center 3: 3 = 3 basis function, range (10, 11)
    basis.add(forte2.ints.Shell(1, [1.0], [1.0], [1.0, 0.0, 0.0]))
    print(basis.center_first_and_last)
    assert basis.center_first_and_last == [(0, 4), (4, 10), (10, 13)]


if __name__ == "__main__":
    test_basis_center_first_and_last()
