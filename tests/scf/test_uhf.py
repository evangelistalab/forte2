import forte2
import numpy as np
import scipy as sp

from forte2.scf import UHF, RHF


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

    scf = UHF(charge=0, ms=1)(system)
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

    scf = UHF(charge=0, ms=0)(system)
    scf.run()
    assert np.isclose(
        scf.E, euhf, atol=1e-10
    ), f"SCF energy {scf.E} is not close to expected value {euhf}"
    assert np.isclose(
        scf.S2, s2uhf, atol=1e-10
    ), f"SCF S2 {scf.S2} is not close to expected value {s2uhf}"


def test_uhf_one_electron():
    euhf = -0.601864064744
    s2uhf = 0.75
    xyz = """
    H           0.000000000000     0.000000000000     0.000000000000
    H           0.000000000000     0.000000000000     1.000000000000
    """
    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    scf = UHF(charge=1, ms=-0.5)(system)
    scf.econv = 1e-10
    scf.dconv = 1e-8
    scf.run()
    assert np.isclose(
        scf.E, euhf, atol=1e-10, rtol=1e-6
    ), f"SCF energy {scf.E} is not close to expected value {euhf}"
    assert np.isclose(
        scf.S2, s2uhf, atol=1e-10, rtol=1e-6
    ), f"SCF S2 {scf.S2} is not close to expected value {s2uhf}"


def test_coulson_fischer():
    erhf = -0.854139014387
    euhf = -1.000297175136
    s2uhf = 0.987426195958
    xyz = """
    H 0 0 0
    H 0 0 2.7"""
    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    rhf = RHF(charge=0)(system)
    rhf.econv = 1e-10
    rhf.dconv = 1e-8
    rhf.run()
    assert np.isclose(
        rhf.E, erhf, atol=1e-10, rtol=1e-6
    ), f"RHF energy {rhf.E} is not close to expected value {erhf}"
    uhf = UHF(charge=0, ms=0)(system)
    uhf.econv = 1e-10
    uhf.dconv = 1e-8
    # This option mixes the homo and lumo of the initial guess, breaking spin symmetry
    uhf.guess_mix = True
    uhf.run()
    assert np.isclose(
        uhf.E, euhf, atol=1e-10, rtol=1e-6
    ), f"UHF energy {uhf.E} is not close to expected value {euhf}"
    assert np.isclose(
        uhf.S2, s2uhf, atol=1e-10, rtol=1e-6
    ), f"UHF S2 {uhf.S2} is not close to expected value {s2uhf}"


if __name__ == "__main__":
    # test_uhf_triplet()
    # test_uhf_singlet()
    # test_uhf_one_electron()
    test_coulson_fischer()
