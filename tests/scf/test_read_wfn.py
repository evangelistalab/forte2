import forte2
import numpy as np
import scipy as sp
import copy

from forte2.scf import RHF


def test_read_wfn():
    # Test the RHF implementation with a simple example
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system1 = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf1 = RHF(system1, charge=0)
    scf1.run()
    assert np.isclose(
        scf1.E, -76.0614664043887672, atol=1e-10, rtol=1e-8
    ), f"SCF energy {scf1.E} is not close to expected value -76.0614664043887672"
    c0 = copy.deepcopy(scf1.C)

    e_newgeom = -76.061609399548
    xyz = """
    O            0.000000000000     0.000000000000    -0.063
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system2 = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    scf2 = RHF(system2, charge=0)
    scf2.run()
    assert np.isclose(
        scf2.E, e_newgeom, atol=1e-10, rtol=1e-8
    ), f"SCF energy {scf2.E} is not close to expected value {e_newgeom}"

    scf2.C = c0
    scf2.run()
    assert np.isclose(
        scf2.E, e_newgeom, atol=1e-10, rtol=1e-8
    ), f"SCF energy {scf2.E} is not close to expected value {e_newgeom}"


if __name__ == "__main__":
    test_read_wfn()
