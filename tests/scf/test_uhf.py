import forte2
import numpy as np
import scipy as sp

from forte2.scf import UHF


def test_uhf_triplet():
    # Test the UHF implementation with a simple example
    euhf = -75.810772016719
    s2uhf = 2.005739313683
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = UHF(system, charge=0, ms=1)
    scf.run()
    assert np.isclose(
        scf.E, euhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {euhf}"
    assert np.isclose(
        scf.S2, s2uhf, atol=1e-10
    ), f"SCF S2 {scf.S2} is not close to expected value {s2uhf}"


def test_uhf_singlet():
    # Test the UHF implementation with a simple example
    euhf = -76.0614664043887672
    s2uhf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = UHF(system, charge=0, ms=0)
    scf.run()
    assert np.isclose(
        scf.E, euhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {euhf}"
    assert np.isclose(
        scf.S2, s2uhf, atol=1e-10
    ), f"SCF S2 {scf.S2} is not close to expected value {s2uhf}"


if __name__ == "__main__":
    test_uhf_triplet()
    test_uhf_singlet()
