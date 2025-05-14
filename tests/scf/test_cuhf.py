import forte2
import numpy as np
import scipy as sp
import time

from forte2.scf import CUHF


def test_cuhf_singlet():
    # Test the CUHF implementation with a simple example (this is equivalent to RHF)
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = CUHF(charge=0, mult=1)
    scf.run(system)
    assert np.isclose(
        scf.E, -76.0614664043887672, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -76.0614664043887672"


def test_cuhf_triplet():
    # Test the CUHF implementation with a simple example (this is equivalent to ROHF)
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = CUHF(charge=0, mult=3)
    scf.run(system)
    assert np.isclose(
        scf.E, -75.8051090240099, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -75.8051090240099"


if __name__ == "__main__":
    test_cuhf_singlet()
    test_cuhf_triplet()
