import forte2
from forte2.scf import UHF, RHF
import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=1e-8, abs=5e-8)


def test_uhf_triplet():
    # Test the UHF implementation with a simple example
    euhf = -75.810772399321
    s2uhf = 2.005739321130
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, ms=1)(system)
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_uhf_singlet():
    # Test the UHF implementation with a simple example
    euhf = -76.061466407194
    s2uhf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = UHF(charge=0, ms=0)(system)
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_uhf_one_electron():
    euhf = -0.601864064744
    s2uhf = 0.75
    xyz = """
    H           0.000000000000     0.000000000000     0.000000000000
    H           0.000000000000     0.000000000000     1.000000000000
    """
    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    scf = UHF(charge=1, ms=-0.5)(system)
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_coulson_fischer():
    erhf = -0.854139014387
    euhf = -1.000297175136
    s2uhf = 0.987426195958
    xyz = """
    H 0 0 0
    H 0 0 2.7"""
    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    rhf = RHF(charge=0)(system)
    rhf.run()
    assert rhf.E == approx(erhf)

    uhf = UHF(charge=0, ms=0)(system)
    # This option mixes the homo and lumo of the initial guess, breaking spin symmetry
    uhf.guess_mix = True
    uhf.run()
    assert uhf.E == approx(euhf)
    assert uhf.S2 == approx(s2uhf)
