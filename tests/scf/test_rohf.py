import pytest

from forte2 import System
from forte2.scf import ROHF
from forte2.helpers.comparisons import approx


def test_rohf_singlet():
    # Test the ROHF implementation with a simple example (this is equivalent to RHF)
    erohf = -76.061466407194
    s2rohf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = ROHF(charge=0, ms=0)(system)
    scf.run()
    assert scf.E == approx(erohf)
    assert scf.S2 == approx(s2rohf)


def test_rohf_triplet():
    erohf = -75.805109024040
    s2rohf = 2.0

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = ROHF(charge=0, ms=1)(system)
    scf.run()
    assert scf.E == approx(erohf)
    assert scf.S2 == approx(s2rohf)


def test_rohf_incompatible_params():
    xyz = """
    H 0 0 0
    H 0 0 1
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="def2-universal-jkfit"
    )
    with pytest.raises(ValueError):
        scf = ROHF(charge=1)(system)
    with pytest.raises(ValueError):
        scf = ROHF(charge=-3, ms=0)(system)
    with pytest.raises(ValueError):
        scf = ROHF(charge=1, ms=1.0)(system)
    scf = ROHF(charge=-5, ms=0.5)(system)
