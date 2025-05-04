import forte2
import numpy as np
import scipy as sp
import time

from forte2.scf import RHF, UHF, ROHF, CUHF


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

def test_uhf_triplet():
    # Test the UHF implementation with a simple example
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, mult=3)
    scf.run(system)
    assert np.isclose(
        scf.E, -75.8107723993035, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -75.8107723993035"

def test_uhf_singlet():
    # Test the UHF implementation with a simple example
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, mult=1)
    scf.run(system)
    assert np.isclose(
        scf.E, -76.0614664043887672, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -76.0614664043887672"

def test_rohf_singlet():
    # Test the ROHF implementation with a simple example (this is equivalent to RHF)
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, mult=1)
    scf.run(system)
    assert np.isclose(
        scf.E, -76.0614664043887672, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -76.0614664043887672"

def test_rohf_triplet():
    # Test the ROHF implementation with a simple example
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, mult=3)
    scf.run(system)
    assert np.isclose(
        scf.E, -75.8051090240099, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -75.8051090240099"

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
    test_rhf()
    test_uhf_singlet()
    test_uhf_triplet()
    test_rohf_singlet()
    test_rohf_triplet()
    test_cuhf_singlet()
    test_cuhf_triplet()
    print("Test passed.")
