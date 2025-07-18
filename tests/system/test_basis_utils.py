import pytest
import forte2
import numpy as np


def test_basis_info():
    xyz = """
    C 0 0 0
    O 0 0 1.2
    N 0 0 2.4
    O 2 1 0
    """
    system = forte2.System(
        xyz=xyz,
        basis_set={"C": "cc-pvtz", "O": "sto-6g", "default": "sto-3g"},
    )
    basis_info = forte2.basis_utils.BasisInfo(system, system.basis)
    assert basis_info.basis_labels[23] == forte2.basis_utils.BasisInfo._AOLabel(
        23, 0, 6, 1, 4, 3, 0
    )
    assert basis_info.atom_to_aos[8][2] == list(range(40, 45))


def test_get_shell_label():
    assert forte2.basis_utils.get_shell_label(0, 0) == "s"
    assert forte2.basis_utils.get_shell_label(1, 0) == "py"
    assert forte2.basis_utils.get_shell_label(1, 1) == "pz"
    assert forte2.basis_utils.get_shell_label(1, 2) == "px"
    assert forte2.basis_utils.get_shell_label(2, 0) == "dxy"
    assert forte2.basis_utils.get_shell_label(2, 1) == "dyz"
    assert forte2.basis_utils.get_shell_label(2, 2) == "dz2"
    assert forte2.basis_utils.get_shell_label(2, 3) == "dxz"
    assert forte2.basis_utils.get_shell_label(2, 4) == "dx2-y2"
    assert forte2.basis_utils.get_shell_label(3, 0) == "fy(3x2-y2)"
    assert forte2.basis_utils.get_shell_label(3, 1) == "fxyz"
    assert forte2.basis_utils.get_shell_label(3, 2) == "fyz2"
    assert forte2.basis_utils.get_shell_label(3, 3) == "fz3"
    assert forte2.basis_utils.get_shell_label(3, 4) == "fxz2"
    assert forte2.basis_utils.get_shell_label(3, 5) == "fz(x2-y2)"
    assert forte2.basis_utils.get_shell_label(3, 6) == "fx(x2-3y2)"

    assert forte2.basis_utils.get_shell_label(4, 0) == "g(0)"
    assert forte2.basis_utils.get_shell_label(11, 22) == "n(22)"

    with pytest.raises(Exception):
        forte2.basis_utils.get_shell_label(-1, 0)
    with pytest.raises(Exception):
        forte2.basis_utils.get_shell_label(0, -3)
    with pytest.raises(Exception):
        forte2.basis_utils.get_shell_label(0, 4)
    with pytest.raises(Exception):
        forte2.basis_utils.get_shell_label(5, 11)
    with pytest.raises(Exception):
        forte2.basis_utils.get_shell_label(12, 2)


def test_shell_label_to_lm():
    assert forte2.basis_utils.shell_label_to_lm("s") == [(0, 0)]
    assert forte2.basis_utils.shell_label_to_lm("p") == [(1, 0), (1, 1), (1, 2)]
    assert forte2.basis_utils.shell_label_to_lm("i") == [(6, i) for i in range(13)]
    assert forte2.basis_utils.shell_label_to_lm("dxy") == [(2, 0)]
    assert forte2.basis_utils.shell_label_to_lm("dyz") == [(2, 1)]
    assert forte2.basis_utils.shell_label_to_lm("dz2") == [(2, 2)]
    assert forte2.basis_utils.shell_label_to_lm("dxz") == [(2, 3)]
    assert forte2.basis_utils.shell_label_to_lm("dx2-y2") == [(2, 4)]
    assert forte2.basis_utils.shell_label_to_lm("fy(3x2-y2)") == [(3, 0)]
    assert forte2.basis_utils.shell_label_to_lm("fxyz") == [(3, 1)]
    assert forte2.basis_utils.shell_label_to_lm("fyz2") == [(3, 2)]
    assert forte2.basis_utils.shell_label_to_lm("fz3") == [(3, 3)]
    assert forte2.basis_utils.shell_label_to_lm("fxz2") == [(3, 4)]
    assert forte2.basis_utils.shell_label_to_lm("fz(x2-y2)") == [(3, 5)]
    assert forte2.basis_utils.shell_label_to_lm("fx(x2-3y2)") == [(3, 6)]
    assert forte2.basis_utils.shell_label_to_lm("g(0)") == [(4, 0)]
    assert forte2.basis_utils.shell_label_to_lm("n(22)") == [(11, 22)]

    with pytest.raises(Exception):
        forte2.basis_utils.shell_label_to_lm("sx")
    with pytest.raises(Exception):
        forte2.basis_utils.shell_label_to_lm("dyx")
    with pytest.raises(Exception):
        forte2.basis_utils.shell_label_to_lm("d2")
    with pytest.raises(Exception):
        forte2.basis_utils.shell_label_to_lm("g(-1)")
    with pytest.raises(Exception):
        forte2.basis_utils.shell_label_to_lm("h(11)")
