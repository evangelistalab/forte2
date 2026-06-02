import pytest
import numpy as np

from forte2 import System
from forte2.system.basis_utils import BasisInfo, get_shell_label, shell_label_to_lm
from forte2.system.build_basis import decontract_basis, build_basis, BSE_AVAILABLE
from forte2.integrals import overlap


def test_basis_info():
    xyz = """
    C 0 0 0
    O 0 0 1.2
    N 0 0 2.4
    O 2 1 0
    """
    system = System(
        xyz=xyz,
        basis_set={"C": "cc-pvtz", "O": "sto-6g", "default": "sto-3g"},
    )
    basis_info = BasisInfo(system, system.basis)
    assert basis_info.basis_labels[23] == BasisInfo._AOLabel(23, 0, 6, 1, 4, 3, -3, 0)
    assert basis_info.atom_to_aos[8][2] == list(range(40, 45))


def test_basis_serialize():
    xyz = """
    C 0 0 0
    O 0 0 1.2
    N 0 0 2.4
    O 2 1 0
    """
    system = System(
        xyz=xyz,
        basis_set={"C": "cc-pvtz", "O": "sto-6g", "default": "sto-3g"},
    )
    res = system.basis.serialize()

    # this test needs to be updated when the schema version is updated
    assert res["schema_version"] == 1
    assert res["nshells"] == 19
    shells = res["shells"]
    for i in range(system.basis.nshells):
        sh = system.basis[i]
        sh_json = shells[i]
        assert sh_json["nprim"] == sh.nprim
        assert sh_json["l"] == sh.l
        assert sh_json["exponents"] == pytest.approx(sh.exponents, abs=1e-10)
        assert sh_json["coefficients"] == pytest.approx(sh.coeff, abs=1e-10)
        assert sh_json["center"] == pytest.approx(sh.center, abs=1e-10)


def test_get_shell_label():
    assert get_shell_label(0, 0) == "s"
    assert get_shell_label(1, 0) == "py"
    assert get_shell_label(1, 1) == "pz"
    assert get_shell_label(1, 2) == "px"
    assert get_shell_label(2, 0) == "dxy"
    assert get_shell_label(2, 1) == "dyz"
    assert get_shell_label(2, 2) == "dz2"
    assert get_shell_label(2, 3) == "dxz"
    assert get_shell_label(2, 4) == "dx2-y2"
    assert get_shell_label(3, 0) == "fy(3x2-y2)"
    assert get_shell_label(3, 1) == "fxyz"
    assert get_shell_label(3, 2) == "fyz2"
    assert get_shell_label(3, 3) == "fz3"
    assert get_shell_label(3, 4) == "fxz2"
    assert get_shell_label(3, 5) == "fz(x2-y2)"
    assert get_shell_label(3, 6) == "fx(x2-3y2)"

    assert get_shell_label(4, 0) == "g(0)"
    assert get_shell_label(11, 22) == "n(22)"

    with pytest.raises(Exception):
        get_shell_label(-1, 0)
    with pytest.raises(Exception):
        get_shell_label(0, -3)
    with pytest.raises(Exception):
        get_shell_label(0, 4)
    with pytest.raises(Exception):
        get_shell_label(5, 11)
    with pytest.raises(Exception):
        get_shell_label(12, 2)


def test_shell_label_to_lm():
    assert shell_label_to_lm("s") == [(0, 0)]
    assert shell_label_to_lm("p") == [(1, 0), (1, 1), (1, 2)]
    assert shell_label_to_lm("i") == [(6, i) for i in range(13)]
    assert shell_label_to_lm("dxy") == [(2, 0)]
    assert shell_label_to_lm("dyz") == [(2, 1)]
    assert shell_label_to_lm("dz2") == [(2, 2)]
    assert shell_label_to_lm("dxz") == [(2, 3)]
    assert shell_label_to_lm("dx2-y2") == [(2, 4)]
    assert shell_label_to_lm("fy(3x2-y2)") == [(3, 0)]
    assert shell_label_to_lm("fxyz") == [(3, 1)]
    assert shell_label_to_lm("fyz2") == [(3, 2)]
    assert shell_label_to_lm("fz3") == [(3, 3)]
    assert shell_label_to_lm("fxz2") == [(3, 4)]
    assert shell_label_to_lm("fz(x2-y2)") == [(3, 5)]
    assert shell_label_to_lm("fx(x2-3y2)") == [(3, 6)]
    assert shell_label_to_lm("g(0)") == [(4, 0)]
    assert shell_label_to_lm("n(22)") == [(11, 22)]

    with pytest.raises(Exception):
        shell_label_to_lm("sx")
    with pytest.raises(Exception):
        shell_label_to_lm("dyx")
    with pytest.raises(Exception):
        shell_label_to_lm("d2")
    with pytest.raises(Exception):
        shell_label_to_lm("g(-1)")
    with pytest.raises(Exception):
        shell_label_to_lm("h(11)")


def test_basis_info_spinor():
    system = System(
        xyz="C 0 0 0",
        basis_set="cc-pvdz",
    )
    basis_info = BasisInfo(system, system.basis)
    assert basis_info.basis_labels_spinor[22] == BasisInfo._SpinorAOLabel(
        22, 0, 6, 1, 3, 2, 5, -5
    )


@pytest.mark.skipif(not BSE_AVAILABLE, reason="Basis set exchange is not available")
def test_decontract():
    xyz = """
    C 0 0 0
    O 0 0 1
    H 0 0 2
    O 1 0 1
    """
    basis_set = {
        "C": "cc-pvdz",
        "O1": "sto-6g",
        "O2": "def2-svp",
        "default": "cc-pvtz",
    }
    system = System(xyz=xyz, basis_set=basis_set)
    basis_decon1 = build_basis(basis_set, system.geom_helper, decontract=True)
    basis_decon2 = decontract_basis(system.basis)
    assert len(basis_decon1) == len(basis_decon2)
    ovlp1 = overlap(system, basis_decon1)
    ovlp_cross = overlap(system, basis_decon1, basis_decon2)
    ovlp2 = overlap(system, basis_decon2)
    # S22 = S21 @ S11^{-1} @ S12
    ovlp2_recon = ovlp_cross.T @ np.linalg.solve(ovlp1, ovlp_cross)
    assert np.linalg.norm(ovlp2 - ovlp2_recon) <= 1e-11
