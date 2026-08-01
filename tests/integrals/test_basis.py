import forte2


def test_basis_properties():
    # Test the Basis class
    basis = forte2.ints.Basis()
    # center 1: 1 + 3 = 4 basis functions, range (0, 4)
    basis.add(forte2.ints.Shell(0, [1.0], [1.0], [0.0, 0.0, 0.0]))
    basis.add(forte2.ints.Shell(1, [1.0], [1.0], [0.0, 0.0, 0.0]))
    # center 2: 1 + 5 = 6 basis functions, range (4, 10)
    basis.add(forte2.ints.Shell(0, [1.0, 2.0], [1.0, 0.5], [0.0, 0.0, 1.0]))
    basis.add(forte2.ints.Shell(2, [1.0], [1.0], [0.0, 0.0, 1.0]))
    # center 3: 3 = 3 basis function, range (10, 13)
    basis.add(forte2.ints.Shell(1, [1.0], [1.0], [1.0, 0.0, 0.0]))
    assert basis.center_first_and_last == [(0, 4), (4, 10), (10, 13)]
    assert len(basis) == 13
    assert basis.max_l == 2
    assert basis.max_nprim == 2
    assert basis.max_nbasis == 5


def test_shell_label_out_of_range_raises():
    # Regression: shell_label validated l < 0 but not the upper bound, so
    # general_labels[l] was an out-of-bounds vector read (UB) for l beyond the
    # defined labels (l >= 12).
    import pytest

    # Defined labels: explicit s..f, general s..n (l = 0..11).
    assert forte2.ints.shell_label(11, 0) == "n(0)"
    assert forte2.ints.shell_label(4, 0) == "g(0)"
    assert forte2.ints.shell_label(2, 0) == "dxy"

    with pytest.raises(Exception):
        forte2.ints.shell_label(12, 0)
    with pytest.raises(Exception):
        forte2.ints.shell_label(-1, 0)
