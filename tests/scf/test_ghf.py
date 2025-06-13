import forte2
from forte2.scf import *
import numpy as np
from forte2.helpers.comparisons import approx


def test_equivalence_to_rhf():
    e_ghf = -128.488756188998
    s2_ghf = 0.0
    xyz = """
    Ne 0 0 0
    """

    system = forte2.System(
        xyz=xyz, basis="cc-pvdz", auxiliary_basis="def2-universal-jkfit"
    )

    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(e_ghf)
    assert scf.S2 == approx(s2_ghf)


def test_equivalence_to_uhf():
    # This is the same test as test_uhf_coulson_fischer
    euhf = -1.000297175136
    s2uhf = 0.987426195959
    xyz = """
    H 0 0 0
    H 0 0 2.7"""
    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    scf = GHF(charge=0)(system)
    scf.guess_mix = True
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_ghf():
    e_ghf = -75.649277913857
    s2_ghf = 0.756178428699

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pvqz", auxiliary_basis="cc-pvqz-jkfit")

    scf = GHF(charge=1)(system)
    scf.run()
    assert scf.E == approx(e_ghf)
    assert scf.S2 == approx(s2_ghf)


def test_break_complex_symmetry():
    eghf_real = -1.514563104178
    s2ghf_real = 0.770638666820
    eghf = -1.516054958839
    s2ghf = 0.776532390590

    xyz = f"""
    H 0 0 0
    H 1 0 0
    H 0.5 {0.5*np.sqrt(3)} 0
    """

    system = forte2.System(xyz=xyz, basis="cc-pvqz", auxiliary_basis="cc-pvqz-jkfit")

    scf = GHF(charge=0)(system)
    scf.break_complex_symmetry = False
    # This is a case where the minao initial gets a worse energy than the hcore guess.
    scf.guess_type = "hcore"
    scf.run()
    assert scf.E == approx(eghf_real)
    assert scf.S2 == approx(s2ghf_real)

    scf.break_complex_symmetry = True
    # Do not use the previous C as a guess
    scf.C = None
    scf.run()
    assert scf.E == approx(eghf)
    assert scf.S2 == approx(s2ghf)

    scf.break_complex_symmetry = False
    # Automatically uses the previous C as a guess
    scf.run()
    assert scf.E == approx(eghf)
    assert scf.S2 == approx(s2ghf)
