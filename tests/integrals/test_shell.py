import pytest
import forte2


def test_shell():
    # Test the Shell class
    shell = forte2.ints.Shell(0, [1.0], [1.0], [0.0, 0.0, 0.0])
    assert shell.size == 1
    assert shell.nprim == 1
    assert shell.ncontr == 1


if __name__ == "__main__":
    test_shell()
