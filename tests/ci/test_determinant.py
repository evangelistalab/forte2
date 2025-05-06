import pytest
import forte2


def test_determinant():
    # Test the determinant class initialization with the zero static method
    d = forte2.Determinant.zero()

    for i in range(1, 64):
        assert d.get_a(i) == False
        assert d.get_b(i) == False

    assert (
        str(d) == "|0000000000000000000000000000000000000000000000000000000000000000>"
    )

    assert d.count() == 0
    assert d.count_a() == 0
    assert d.count_b() == 0

    # Test the determinant class initialization with a string
    d = forte2.Determinant("")

    for i in range(1, 64):
        assert d.get_a(i) == False
        assert d.get_b(i) == False

    assert (
        str(d) == "|0000000000000000000000000000000000000000000000000000000000000000>"
    )

    assert d.count() == 0
    assert d.count_a() == 0
    assert d.count_b() == 0


def test_determinant_set_get():
    # Test the determinant class set and get methods
    d = forte2.Determinant.zero()
    for i in range(1, 64):
        assert d.get_a(i) == False
        assert d.get_b(i) == False

    set_a = [1, 2, 3, 4, 5, 63]
    set_b = [6, 7, 8, 9, 10]
    for i in set_a:
        d.set_a(i, True)
    for i in set_b:
        d.set_b(i, True)

    # Test the determinant class get methods after setting values
    for i in range(64):
        assert d.get_a(i) == (i in set_a)
        assert d.get_b(i) == (i in set_b)

    assert d.count() == len(set_a) + len(set_b)
    assert d.count_a() == len(set_a)
    assert d.count_b() == len(set_b)

    # Test the determinant copy constructor
    d2 = forte2.Determinant(d)
    for i in range(64):
        assert d2.get_a(i) == (i in set_a)
        assert d2.get_b(i) == (i in set_b)


if __name__ == "__main__":
    test_determinant()
