import forte2
import numpy as np
import scipy as sp

from forte2.scf import CUHF


def test_cuhf_singlet():
    # Test the CUHF implementation with a simple example (this is equivalent to RHF)
    ecuhf = -76.0614664043887672
    s2cuhf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = CUHF(system, charge=0, ms=0)
    scf.run()
    assert np.isclose(
        scf.E, ecuhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {ecuhf}"
    assert np.isclose(
        scf.S2, s2cuhf, atol=1e-10
    ), f"SCF S2 {scf.S2} is not close to expected value {s2cuhf}"


def test_cuhf_triplet():
    # Test the CUHF implementation with a simple example (this is equivalent to ROHF)
    ecuhf = -75.8051090240099
    s2cuhf = 2.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = CUHF(charge=0, ms=1)(system)
    scf.run()
    assert np.isclose(
        scf.E, ecuhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {ecuhf}"
    assert np.isclose(
        scf.S2, s2cuhf, atol=1e-10
    ), f"SCF S2 {scf.S2} is not close to expected value {s2cuhf}"


if __name__ == "__main__":
    test_cuhf_singlet()
    test_cuhf_triplet()
