import forte2
from forte2.scf import ROHF
import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=0.0, abs=5e-8)


def test_rohf_singlet():
    # Test the ROHF implementation with a simple example (this is equivalent to RHF)
    erohf = -76.061466407194
    s2rohf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

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

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = ROHF(charge=0, ms=1)(system)
    scf.run()
    assert scf.E == approx(erohf)
    assert scf.S2 == approx(s2rohf)


if __name__ == "__main__":
    test_rohf_singlet()
    test_rohf_triplet()
