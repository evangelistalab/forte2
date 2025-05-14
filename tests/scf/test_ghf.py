import forte2
import numpy as np
import scipy as sp
import time

from forte2.scf import GHF


def test_ghf():
    # Test the RHF implementation with a simple example
    e_ghf = -128.48875618899837
    xyz = """
    Ne 0 0 0
    """

    system = forte2.System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit"
    )

    scf = GHF(charge=0, mult=1)
    scf.run(system, econv=1e-8, dconv=1e-6)
    assert np.isclose(
        scf.E, e_ghf, atol=1e-6
    ), f"RHF energy mismatch: {scf.E} vs {e_ghf}"


def test_ghf2():
    # Test the RHF implementation with a simple example
    e_ghf = -75.64927791393633
    s2_ghf = 0.756178428697

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pvqz", auxiliary_basis="cc-pvqz-jkfit")

    scf = GHF(charge=1, mult=2)
    scf.run(system, econv=1e-10, dconv=1e-8)
    assert np.isclose(
        scf.E, e_ghf, atol=1e-6
    ), f"GHF energy mismatch: {scf.E} vs {e_ghf}"


def test_ghf3():

    xyz = f"""
    H 0 0 0
    H 1 0 0
    H 0.5 {0.5*np.sqrt(3)} 0
    """

    system = forte2.System(xyz=xyz, basis="cc-pvqz", auxiliary_basis="cc-pvqz-jkfit")

    scf = GHF(charge=0, mult=2)
    scf.run(system, econv=1e-10, dconv=1e-8)


if __name__ == "__main__":
    test_ghf()
    test_ghf2()
    test_ghf3()
