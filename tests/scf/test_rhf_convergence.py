import pytest

from forte2 import System, RHF
from forte2.helpers.comparisons import approx
from forte2.system import BSE_AVAILABLE


@pytest.mark.skipif(not BSE_AVAILABLE, reason="BSE not available")
def test_cadmium_imidazole_complex(tmp_path):
    eref = -5735.181493863483
    # Geometry from SI of 10.1063/1.2974099
    # See discussion therein and also in 10.1063/1.3304922
    xyz = """
    Cd       0.000000000    0.000000000     0.000000000 
    N        0.000000000    0.000000000    -4.270782744
    N       -1.295300812    0.000000000    -8.216595657
    C        1.277555548    0.000000000    -8.286579686
    C        2.050807130    0.000000000    -5.841580898
    C       -1.974295739    0.000000000    -5.782977653
    H        2.327254176    0.000000000   -10.016981910
    H        3.946959286    0.000000000    -5.123193109
    H       -3.909370623    0.000000000    -5.152366230
    H       -2.481531661    0.000000000    -9.778799182
    """
    system = System(
        xyz=xyz,
        basis_set="3-21g",
        auxiliary_basis_set="def2-universal-jkfit",
        minao_basis_set=None,
        unit="bohr",
    )
    system.save(tmp_path / "test_cadmium_imidazole_complex")
    system_load = System.load(tmp_path / "test_cadmium_imidazole_complex")
    # explicitly test that basis_set is not needed if system is loaded
    system.basis_set = None

    scf = RHF(charge=2)(system_load)
    scf.run()
    assert scf.E == approx(eref)


def test_rhf_cu():
    eref = -1638.952656289754
    xyz = """
    Cu 0 0 0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    mf = RHF(charge=-1)(system)
    mf.run()
    assert mf.E == approx(eref)
