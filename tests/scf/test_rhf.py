import forte2
import numpy as np
import scipy as sp

from forte2.scf import RHF


def test_rhf():
    # Test the RHF implementation with a simple example
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)
    scf.run(system)
    assert np.isclose(
        scf.E, -76.0614664043887672, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -76.0614664043887672"


if __name__ == "__main__":
    test_rhf()
