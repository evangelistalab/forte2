import pytest
import forte2


def test_shell():
    forte2.Shell()
    print("Shell test passed.")
    # , int l, bool is_pure, const std::vector<double>& exponents,
    #             const std::vector<double>& coeffs, const std::vector<double>& centers
    shell = forte2.Shell(0, [1.0], [1.0], [0.0, 0.0, 0.0])
    assert shell.size == 1
    assert shell.nprim == 1
    assert shell.ncontr == 1


if __name__ == "__main__":
    test_shell()
