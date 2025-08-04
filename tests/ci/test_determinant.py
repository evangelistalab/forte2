from forte2 import Determinant


def test_determinant():
    # Test the determinant class initialization with the zero static method
    d = Determinant.zero()

    for i in range(1, 64):
        assert d.na(i) == False
        assert d.nb(i) == False

    assert (
        str(d) == "|0000000000000000000000000000000000000000000000000000000000000000>"
    )

    assert d.count() == 0
    assert d.count_a() == 0
    assert d.count_b() == 0

    # Test the determinant class initialization with a string
    d = Determinant("")

    for i in range(1, 64):
        assert d.na(i) == False
        assert d.nb(i) == False

    assert (
        str(d) == "|0000000000000000000000000000000000000000000000000000000000000000>"
    )

    assert d.count() == 0
    assert d.count_a() == 0
    assert d.count_b() == 0


def test_determinant_set_get():
    # Test the determinant class set and get methods
    d = Determinant.zero()
    for i in range(1, 64):
        assert d.na(i) == False
        assert d.nb(i) == False

    set_a = [1, 2, 3, 4, 5, 63]
    set_b = [6, 7, 8, 9, 10]
    for i in set_a:
        d.set_na(i, True)
    for i in set_b:
        d.set_nb(i, True)

    # Test the determinant class get methods after setting values
    for i in range(64):
        assert d.na(i) == (i in set_a)
        assert d.nb(i) == (i in set_b)

    assert d.count() == len(set_a) + len(set_b)
    assert d.count_a() == len(set_a)
    assert d.count_b() == len(set_b)

    # Test the determinant copy constructor
    d2 = Determinant(d)
    for i in range(64):
        assert d2.na(i) == (i in set_a)
        assert d2.nb(i) == (i in set_b)


def test_det_equality():
    """Test the __eq__ operator"""
    d1 = Determinant("22")
    d2 = Determinant("2+")
    d3 = Determinant("22")
    d4 = Determinant("0022")
    assert d1 == d1
    assert d1 != d2
    assert d1 == d3
    assert d2 != d4
    assert d1 != d4


def test_det_hash():
    """Test the __hash__ operator"""
    d1 = Determinant("22")
    d2 = Determinant("2+")
    d3 = Determinant("22")
    d4 = Determinant("0022")
    h = {}
    h[d1] = 1.0
    h[d2] = 2.0
    h[d3] += 0.25
    h[d4] = 3.0
    assert h[d1] == 1.25
    assert h[d3] == 1.25
    assert h[d2] == 2.00
    assert h[d4] == 3.00


def test_det_sorting():
    """Test the __lt__ operator"""
    d1 = Determinant("22")
    d2 = Determinant("2+")
    d3 = Determinant("--")
    d4 = Determinant("22")
    list = [d1, d2, d3, d4]
    sorted_list = sorted(list)
    assert sorted_list[0] == d2
    assert sorted_list[1] == d3
    assert sorted_list[2] == d1
    assert sorted_list[3] == d4
    assert sorted_list[2] == d4
    assert sorted_list[3] == d1


def test_gen_excitation():
    # test a -> a excitation
    d1 = Determinant("220")
    assert d1.gen_excitation([0], [3], [], []) == -1.0
    assert d1 == Determinant("-20+")

    # test b -> b excitation
    d2 = Determinant("2-+0")
    assert d2.gen_excitation([], [], [0, 1], [2, 3]) == -1.0
    assert d2 == Determinant("+02-")

    # test b creation and counting number of a
    d3 = Determinant("+000")
    assert d3.gen_excitation([], [], [], [0]) == -1.0
    assert d3 == Determinant("2")
    d3 = Determinant("0000")
    assert d3.gen_excitation([], [], [], [0]) == +1.0
    assert d3 == Determinant("-")

    # test ab creation and sign
    d4 = Determinant("000")
    assert d4.gen_excitation([], [2, 1], [], [0, 1]) == -1.0
    assert d4 == Determinant("-2+")
    d5 = Determinant("000")
    assert d5.gen_excitation([], [2, 1], [], [1, 0]) == +1.0
    assert d5 == Determinant("-2+")
    d6 = Determinant("000")
    assert d6.gen_excitation([], [1, 2], [], [0, 1]) == +1.0
    assert d6 == Determinant("-2+")
    d7 = Determinant("000")
    assert d7.gen_excitation([], [1, 2], [], [1, 0]) == -1.0
    assert d7 == Determinant("-2+")


def test_excitation_connection():
    """Test the excitation_connection function"""
    d1 = Determinant("220")
    d2 = Determinant("022")
    conn = d1.excitation_connection(d2)
    assert conn[0] == [0]  # alfa hole
    assert conn[1] == [2]  # alfa particle
    assert conn[2] == [0]  # beta hole
    assert conn[3] == [2]  # beta particle
    conn = d2.excitation_connection(d1)
    assert conn[0] == [2]  # alfa hole
    assert conn[1] == [0]  # alfa particle
    assert conn[2] == [2]  # beta hole
    assert conn[3] == [0]  # beta particle

    # test different number of electrons
    d1 = Determinant("2")
    d2 = Determinant("0")
    conn = d1.excitation_connection(d2)
    assert conn[0] == [0]  # alfa hole
    assert conn[1] == []  # alfa particle
    assert conn[2] == [0]  # beta hole
    assert conn[3] == []  # beta particle
    conn = d2.excitation_connection(d1)
    assert conn[0] == []  # alfa hole
    assert conn[1] == [0]  # alfa particle
    assert conn[2] == []  # beta hole
    assert conn[3] == [0]  # beta particle

    d1 = Determinant("222+-00000")
    d2 = Determinant("-++0200-02")
    conn = d1.excitation_connection(d2)
    assert conn[0] == [0, 3]  # alfa hole
    assert conn[1] == [4, 9]  # alfa particle
    assert conn[2] == [1, 2]  # beta hole
    assert conn[3] == [7, 9]  # beta particle


def test_det_slater_sign():
    """Test Slater sign functions"""

    #        012345
    # parity 011001
    d = Determinant("+0+0++")
    assert d.slater_sign(0) == 1
    assert d.slater_sign(1) == -1
    assert d.slater_sign(2) == -1
    assert d.slater_sign(3) == 1
    assert d.slater_sign(4) == 1
    assert d.slater_sign(5) == -1
    assert d.slater_sign(6) == 1
    assert d.slater_sign(7) == 1
    assert d.slater_sign_reverse(0) == -1
    assert d.slater_sign_reverse(1) == -1
    assert d.slater_sign_reverse(2) == 1
    assert d.slater_sign_reverse(3) == 1
    assert d.slater_sign_reverse(4) == -1
    assert d.slater_sign_reverse(5) == 1
    assert d.slater_sign_reverse(6) == 1


def test_spin_flip():
    """Test spin flip functions"""
    d = Determinant("2-+0+-0++--")
    assert d.spin_flip() == Determinant("2+-0-+0--++")
