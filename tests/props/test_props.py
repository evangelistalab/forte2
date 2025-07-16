import numpy as np
import forte2
from forte2.scf import RHF, UHF
from forte2.helpers.comparisons import approx, approx_loose


def test_core_energy():
    erhf = -76.061466407195
    ecore = -124.11336728113105
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    ke = forte2.get_property(scf, "kinetic_energy")
    ve = forte2.get_property(scf, "nuclear_attraction_energy")
    assert ke + ve == approx(ecore)


def test_dipole_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    dip = forte2.get_property(scf, "dipole")
    print(dip)
    assert dip == approx_loose([0.00000, 0.00000, 1.95868013])


def test_dipole_uhf():
    euhf = -75.649277914372
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = UHF(charge=1, ms=0.5)(system)
    e_dip = forte2.get_property(scf, "electric_dipole")
    # get_property will run the scf if not already run
    assert scf.E == approx(euhf)
    assert e_dip == approx_loose([0, 0, -2.56784946e-02])
    dip = forte2.get_property(scf, "dipole", origin=[1.2, -0.7, 1])
    assert dip == approx([-3.05009553, 1.77922239, -0.23558438])


def test_quadrupole_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    quad = forte2.get_property(scf, "quadrupole")
    print(quad)
    assert np.trace(quad) == approx_loose(0.0)
    assert np.diag(quad) == approx([-2.16502486e00, 2.28286793e00, -1.17843071e-01])


def test_mulliken_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    mp = forte2.mulliken_population(scf)
    assert mp[1] == approx([-0.4620044, 0.2310022, 0.2310022])
