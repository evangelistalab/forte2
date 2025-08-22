from forte2 import System, GHF, UHF
from forte2.helpers.comparisons import approx


def test_equivalence_to_rhf():
    e_ghf = -128.488756188998
    s2_ghf = 0.0
    xyz = """
    Ne 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
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
    system = System(
        xyz=xyz,
        basis_set="cc-pVQZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
    )
    scf = GHF(charge=0, ms_guess=0.0)(system)
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

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvqz-jkfit",
    )

    scf = UHF(charge=1, ms=0.5)(system)
    scf.run()
    assert scf.E == approx(e_uhf)
    assert scf.S2 == approx(s2_uhf)

    scf = GHF(charge=1)(system)
    # this option breaks Sz in the initial guess
    scf.alpha_beta_mix = True
    # this option breaks time-reversal/complex conjugation symmetry in the initial guess
    scf.break_complex_symmetry = True
    scf.run()
    assert scf.E == approx(e_uhf)
    assert scf.S2 == approx(s2_uhf)


def test_j_adapted_ghf():
    # The two bases should yield the same result
    eref = -75.427367675651
    s2ref = 0.7525463566917241
    xyz = """
    O 0 0 0
    H 0 0 1.1"""
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c_type="so",
    )
    scf = GHF(charge=0, j_adapt=False)(system)
    scf.run()
    assert scf.E == approx(eref)
    assert scf.S2 == approx(s2ref)

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c_type="so",
    )
    scf = GHF(charge=0, j_adapt=True)(system)
    scf.run()
    assert scf.E == approx(eref)
    assert scf.S2 == approx(s2ref)


def test_equivalence_to_high_spin_uhf():
    euhf = -37.686541301113
    s2uhf = 2.0063122057868483
    xyz = """
    C 0 0 0"""
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    scf = GHF(charge=0, ms_guess=1.0)(system)
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)
