import forte2
import numpy as np
import scipy as sp

from forte2.scf import GHF


def test_ghf1():
    e_ghf = -128.48875618899837
    s2_ghf = 0.0
    xyz = """
    Ne 0 0 0
    """

    system = forte2.System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit"
    )

    scf = GHF(charge=0)(system)
    scf.econv = 1e-8
    scf.dconv = 1e-6
    scf.run()
    assert np.isclose(
        scf.E, e_ghf, atol=1e-8, rtol=1e-6
    ), f"RHF energy mismatch: {scf.E} vs {e_ghf}"
    assert np.isclose(
        scf.S2, s2_ghf, atol=1e-8, rtol=1e-6
    ), f"RHF S2 mismatch: {scf.S2} vs {s2_ghf}"


def test_ghf2():
    e_ghf = -75.64927791393633
    s2_ghf = 0.756178428697

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pvqz", auxiliary_basis="cc-pvqz-jkfit")

    scf = GHF(charge=1)(system)
    scf.econv = 1e-10
    scf.dconv = 1e-8
    scf.run()
    assert np.isclose(
        scf.E, e_ghf, atol=1e-8, rtol=1e-6
    ), f"GHF energy mismatch: {scf.E} vs {e_ghf}"
    assert np.isclose(
        scf.S2, s2_ghf, atol=1e-8, rtol=1e-6
    ), f"GHF S2 mismatch: {scf.s2} vs {s2_ghf}"


def test_ghf3():
    eghf_real = -1.513661163386
    s2ghf_real = 0.755337181051
    eghf = -1.516054958886
    s2ghf = 0.776532390615

    xyz = f"""
    H 0 0 0
    H 1 0 0
    H 0.5 {0.5*np.sqrt(3)} 0
    """

    system = forte2.System(xyz=xyz, basis="cc-pvqz", auxiliary_basis="cc-pvqz-jkfit")

    scf = GHF(charge=0)(system)
    scf.econv = 1e-10
    scf.dconv = 1e-8
    scf.break_spin_symmetry = False
    scf.break_complex_symmetry = False
    scf.run()

    assert np.isclose(
        scf.E, eghf_real, atol=1e-8, rtol=1e-6
    ), f"GHF energy mismatch: {scf.E} vs {eghf_real}"
    assert np.isclose(
        scf.S2, s2ghf_real, atol=1e-8, rtol=1e-6
    ), f"GHF S2 mismatch: {scf.S2} vs {s2ghf_real}"

    scf.break_spin_symmetry = True
    scf.break_complex_symmetry = True
    scf.C = None
    scf.run()
    assert np.isclose(
        scf.E, eghf, atol=1e-8, rtol=1e-6
    ), f"GHF energy mismatch: {scf.E} vs {eghf}"
    assert np.isclose(
        scf.S2, s2ghf, atol=1e-8, rtol=1e-6
    ), f"GHF S2 mismatch: {scf.S2} vs {s2ghf}"


if __name__ == "__main__":
    test_ghf1()
    test_ghf2()
    test_ghf3()
