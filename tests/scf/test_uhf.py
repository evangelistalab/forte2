import pytest
import forte2
from forte2.scf import UHF, RHF
from forte2.helpers.comparisons import approx


def test_uhf_triplet():
    # Test the UHF implementation with a simple example
    euhf = -75.810772399321
    s2uhf = 2.005739321130
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

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

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

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
    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )
    scf = UHF(charge=1, ms=-0.5)(system)
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_uhf_imcompatible_params():
    xyz = """
    H 0 0 0
    H 0 0 1
    """
    system = forte2.System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="def2-universal-jkfit"
    )
    with pytest.raises(ValueError):
        scf = UHF(charge=-3, ms=1.0)(system)  # 5 electron, ms=1.0 should not be allowed
    with pytest.raises(ValueError):
        scf = UHF(charge=0)(system)  # ms must be explicitly set for UHF
    scf = UHF(charge=-3, ms=0.5)(system)


def test_coulson_fischer():
    erhf = -0.854139014387
    euhf = -1.000297175136
    s2uhf = 0.987426195958
    xyz = """
    H 0 0 0
    H 0 0 2.7"""
    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )
    rhf = RHF(charge=0)(system)
    rhf.run()
    assert rhf.E == approx(erhf)

    uhf = UHF(charge=0, ms=0)(system)
    # This option mixes the homo and lumo of the initial guess, breaking spin symmetry
    uhf.guess_mix = True
    uhf.run()
    assert uhf.E == approx(euhf)
    assert uhf.S2 == approx(s2uhf)
