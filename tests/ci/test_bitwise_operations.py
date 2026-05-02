import forte2


def test_ui64_find_highest_one_bit_all_ones_word():
    uint64_max = (1 << 64) - 1

    assert forte2.ui64_find_highest_one_bit(0) == uint64_max
    assert forte2.ui64_find_highest_one_bit(uint64_max) == 63
    assert forte2.ui64_find_highest_one_bit(1 << 42) == 42
