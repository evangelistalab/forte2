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

    scf = RHF(charge=0)(system)
    scf.run()
    assert np.isclose(
        scf.E, -76.0614664043887672, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value -76.0614664043887672"

def test_rhf_zero_electron():
    xyz = """
    H           0.000000000000     0.000000000000     0.000000000000
    H           0.000000000000     0.000000000000     1.000000000000
    """
    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    scf = RHF(charge=2)(system)
    scf.econv = 1e-10
    scf.run()
    assert np.isclose(
        scf.E, system.nuclear_repulsion_energy(), atol=1e-10, rtol=1e-8,
    ), f"SCF energy {scf.E} is not close to expected value {system.nuclear_repulsion_energy()}"

def test_rhf_zero_virtuals():
    erhf = -126.60457333961503
    xyz = 'Ne 0 0 0'
    system = forte2.System(xyz=xyz, basis="sto-3g", auxiliary_basis="def2-universal-JKFIT")
    scf = RHF(charge=0)(system)
    scf.econv = 1e-10
    scf.run()
    assert np.isclose(
        scf.E, erhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {erhf}"


if __name__ == "__main__":
    test_rhf()
    test_rhf_zero_electron()
    test_rhf_zero_virtuals()