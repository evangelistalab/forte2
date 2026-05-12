from forte2 import Determinant

PARITY_ALPHA_OCC = (0, 1, 5, 17, 62, 63)
PARITY_BETA_OCC = (0, 3, 11, 31, 62, 63)
PARITY_TEST_INDICES = (0, 1, 5, 17, 62, 63, 64, 65, 67, 95, 126, 127)
PAIR_ALPHA_OCC = (0, 1, 5, 17, 31, 62, 63)
PAIR_BETA_OCC = (0, 3, 11, 31, 47, 62, 63)


def _det_with_occupations(alpha_occ, beta_occ):
    d = Determinant.zero()
    for i in alpha_occ:
        d.set_na(i, True)
    for i in beta_occ:
        d.set_nb(i, True)
    return d


def _spin_occupation(d, i):
    if i < Determinant.maxnorb:
        return d.na(i)
    return d.nb(i - Determinant.maxnorb)


def _expected_interval_sign(occ, n, m):
    lo = min(n, m)
    hi = max(n, m)
    count = sum(i in occ for i in range(lo + 1, hi))
    return 1 if count % 2 == 0 else -1


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


def test_determinant_find_last_occupation_zero_sectors_return_ui64_bit_not_found():
    ui64_bit_not_found = (1 << 64) - 1
    d = Determinant.zero()

    assert d.find_last_alpha_occ() == ui64_bit_not_found
    assert d.find_last_beta_occ() == ui64_bit_not_found


def test_det_equality():
    """Test the __eq__ operator"""
    d1 = Determinant("22")
    d2 = Determinant("2a")
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
    d2 = Determinant("2a")
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
    d2 = Determinant("2a")
    d3 = Determinant("bb")
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
    assert d1 == Determinant("b20a")

    # test b -> b excitation
    d2 = Determinant("2ba0")
    assert d2.gen_excitation([], [], [0, 1], [2, 3]) == -1.0
    assert d2 == Determinant("a02b")

    # test b creation and counting number of a
    d3 = Determinant("a000")
    assert d3.gen_excitation([], [], [], [0]) == -1.0
    assert d3 == Determinant("2")
    d3 = Determinant("0000")
    assert d3.gen_excitation([], [], [], [0]) == +1.0
    assert d3 == Determinant("b")

    # test ab creation and sign
    d4 = Determinant("000")
    assert d4.gen_excitation([], [2, 1], [], [0, 1]) == -1.0
    assert d4 == Determinant("b2a")
    d5 = Determinant("000")
    assert d5.gen_excitation([], [2, 1], [], [1, 0]) == +1.0
    assert d5 == Determinant("b2a")
    d6 = Determinant("000")
    assert d6.gen_excitation([], [1, 2], [], [0, 1]) == +1.0
    assert d6 == Determinant("b2a")
    d7 = Determinant("000")
    assert d7.gen_excitation([], [1, 2], [], [1, 0]) == -1.0
    assert d7 == Determinant("b2a")


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

    d1 = Determinant("222ab00000")
    d2 = Determinant("baa0200b02")
    conn = d1.excitation_connection(d2)
    assert conn[0] == [0, 3]  # alfa hole
    assert conn[1] == [4, 9]  # alfa particle
    assert conn[2] == [1, 2]  # beta hole
    assert conn[3] == [7, 9]  # beta particle


def test_det_slater_sign():
    """Test Slater sign functions"""

    #        012345
    # parity 011001
    d = Determinant("a0a0aa")
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


def test_det_slater_sign_matches_naive_parity():
    """Test Slater signs against a direct occupation count."""

    d = _det_with_occupations(PARITY_ALPHA_OCC, PARITY_BETA_OCC)

    for i in PARITY_TEST_INDICES:
        count = sum(_spin_occupation(d, j) for j in range(i))
        expected = 1 if count % 2 == 0 else -1
        assert d.slater_sign(i) == expected


def test_det_slater_sign_reverse_matches_naive_parity():
    """Test reverse Slater signs against a direct occupation count."""

    d = _det_with_occupations(PARITY_ALPHA_OCC, PARITY_BETA_OCC)

    for i in PARITY_TEST_INDICES:
        count = sum(
            _spin_occupation(d, j) for j in range(i + 1, 2 * Determinant.maxnorb)
        )
        expected = 1 if count % 2 == 0 else -1
        assert d.slater_sign_reverse(i) == expected


def test_det_pair_slater_sign_matches_naive_interval_parity():
    """Test pair Slater signs against direct interval occupation counts."""

    d = _det_with_occupations(PAIR_ALPHA_OCC, PAIR_BETA_OCC)

    pairs = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 5),
        (5, 1),
        (0, 63),
        (63, 0),
        (62, 63),
        (17, 62),
        (31, 63),
    ]

    for n, m in pairs:
        assert d.slater_sign_aa(n, m) == _expected_interval_sign(PAIR_ALPHA_OCC, n, m)
        assert d.slater_sign_bb(n, m) == _expected_interval_sign(PAIR_BETA_OCC, n, m)


def test_det_slater_sign_edge_empty_and_full_determinants():
    """Test optimized Slater signs on empty and fully occupied edge cases."""

    empty = Determinant.zero()
    full = _det_with_occupations(range(Determinant.maxnorb), range(Determinant.maxnorb))
    test_indices = (0, 1, 62, 63, 64, 65, 126, 127)

    for i in test_indices:
        assert empty.slater_sign(i) == 1
        assert empty.slater_sign_reverse(i) == 1

        count_before = i
        count_after = 2 * Determinant.maxnorb - i - 1
        assert full.slater_sign(i) == (1 if count_before % 2 == 0 else -1)
        assert full.slater_sign_reverse(i) == (1 if count_after % 2 == 0 else -1)

    for n, m in ((0, 63), (63, 0), (0, 1), (62, 63)):
        expected = _expected_interval_sign(range(Determinant.maxnorb), n, m)
        assert empty.slater_sign_aa(n, m) == 1
        assert empty.slater_sign_bb(n, m) == 1
        assert full.slater_sign_aa(n, m) == expected
        assert full.slater_sign_bb(n, m) == expected


def test_spin_flip():
    """Test spin flip functions"""
    d = Determinant("2ba0ab0aabb")
    assert d.spin_flip() == Determinant("2ab0ba0bbaa")
