from forte2.lib import ints


def test_shell():
    # Test the Shell class
    shell = ints.Shell(0, [1.0], [1.0], [0.0, 0.0, 0.0])
    assert shell.size == 1
    assert shell.nprim == 1
    assert shell.ncontr == 1
