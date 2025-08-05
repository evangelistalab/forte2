import pytest

from forte2 import System
from forte2.scf import CUHF
from forte2.helpers.comparisons import approx


def test_cuhf_singlet():
    # Test the CUHF implementation with a simple example (this is equivalent to RHF)
    ecuhf = -76.061466407194
    s2cuhf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = CUHF(charge=0, ms=0)(system)
    scf.run()
    assert scf.E == approx(ecuhf)
    assert scf.S2 == approx(s2cuhf)


def test_cuhf_triplet():
    # Test the CUHF implementation with a simple example (this is equivalent to ROHF)
    ecuhf = -75.805109024111
    s2cuhf = 2.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = CUHF(charge=0, ms=1)(system)
    scf.run()
    assert scf.E == approx(ecuhf)
    assert scf.S2 == approx(s2cuhf)


def test_cuhf_incompatible_params():
    xyz = """
    H 0 0 0
    H 0 0 1
    """
    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="def2-universal-jkfit"
    )
    with pytest.raises(ValueError):
        scf = CUHF(charge=1)(system)
    with pytest.raises(ValueError):
        scf = CUHF(charge=-3, ms=0)(system)
    with pytest.raises(ValueError):
        scf = CUHF(charge=1, ms=1.0)(system)
    scf = CUHF(charge=-5, ms=0.5)(system)
