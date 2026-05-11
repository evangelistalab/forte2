import forte2


def test_ui64_find_highest_one_bit_edge_cases():
    sentinel = (1 << 64) - 1

    assert forte2.ui64_find_highest_one_bit(0) == sentinel
    assert forte2.ui64_find_highest_one_bit((1 << 64) - 1) == 63
    assert forte2.ui64_find_highest_one_bit(1) == 0
    assert forte2.ui64_find_highest_one_bit(1 << 63) == 63
