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


def test_ghf_complex_perturbation():
    """
    This test checks that, for a system that's stable wrt to
    Sz and time-reversal symmetry breaking, the GHF algorithm will converge to
    the same solution as UHF, even the inital DM breaks Sz and time-reversal symmetries.
    """
    e_uhf = -75.649277913857
    s2_uhf = 0.756178428699

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pvqz", auxiliary_basis="cc-pvqz-jkfit")

    scf = UHF(charge=1, ms=0.5)(system)
    scf.run()
    assert scf.E == approx(e_uhf)
    assert scf.S2 == approx(s2_uhf)

    scf = GHF(charge=1)(system)
    # this option breaks Sz and K symmetries in the initial guess DM
    scf.break_complex_symmetry = True
    scf.run()
    assert scf.E == approx(e_uhf)
    assert scf.S2 == approx(s2_uhf)


def test_break_complex_symmetry():
    """
    Odd regular polygons are prototypical examples of spin-frustrated systems.
    This means that the UHF solution will be unstable wrt Sz and time-reversal
    symmetry breaking
    """
    eghf = -1.514272436189
    s2ghf = 0.777317358363

    xyz = f"""
    H 0 0 0
    H 1 0 0
    H 0.5 {0.5*np.sqrt(3)} 0
    """

    system = forte2.System(xyz=xyz, basis="cc-pvtz", auxiliary_basis="cc-pvqz-jkfit")

    scf = GHF(charge=0, econv=1e-10, dconv=1e-8)(system)
    scf.guess_type = "hcore"
    scf.break_complex_symmetry = True
    scf.run()
    assert scf.E == approx(eghf)
    assert scf.S2 == approx(s2ghf)

    scf.break_complex_symmetry = False
    # Automatically uses the previous C as a guess
    scf.run()
    assert scf.E == approx(eghf)
    assert scf.S2 == approx(s2ghf)
