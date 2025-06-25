import forte2
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

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

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

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = CUHF(charge=0, ms=1)(system)
    scf.run()
    assert scf.E == approx(ecuhf)
    assert scf.S2 == approx(s2cuhf)
