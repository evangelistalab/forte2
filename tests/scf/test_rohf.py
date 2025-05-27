import forte2
import numpy as np
import scipy as sp

from forte2.scf import ROHF


def test_rohf_singlet():
    # Test the ROHF implementation with a simple example (this is equivalent to RHF)
    erohf = -76.0614664043887672
    s2rohf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, ms=0)(system)
    scf.run()
    assert np.isclose(
        scf.E, erohf, atol=1e-10, rtol=1e-6
    ), f"SCF energy {scf.E} is not close to expected value {erohf}"
    assert np.isclose(
        scf.S2, s2rohf, atol=1e-10, rtol=1e-6
    ), f"SCF S2 {scf.S2} is not close to expected value {s2rohf}"


def test_rohf_triplet():
    # Test the ROHF implementation with a simple example
    erohf = -75.8051090240099
    s2rohf = 2.0

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, ms=1)(system)
    scf.run()
    assert np.isclose(
        scf.E, erohf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {erohf}"
    assert np.isclose(
        scf.S2, s2rohf, atol=1e-10
    ), f"SCF S2 {scf.S2} is not close to expected value {s2rohf}"


if __name__ == "__main__":
    test_rohf_singlet()
    test_rohf_triplet()
