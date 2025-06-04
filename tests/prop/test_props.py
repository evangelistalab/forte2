import forte2
from forte2.scf import RHF, UHF
import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=1e-8, abs=5e-8)


def test_core_energy():
    erhf = -76.061466407195
    ecore = -124.11336728113105
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

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

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
    dip = forte2.get_property(scf, "dipole")
    # comparing zero, using default pytest tolerance
    assert dip == pytest.approx([0.00000, 0.00000, 1.95868])


def test_dipole_uhf():
    euhf = -75.649277914372
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")

    scf = UHF(charge=1, ms=0.5)(system)
    scf.run()
    assert scf.E == approx(euhf)
    e_dip = forte2.get_property(scf, "electric_dipole")
    assert e_dip == pytest.approx([0, 0, 1.01026978e-02])
    dip = forte2.get_property(scf, "dipole", origin=[1.2, -0.7, 1])
    assert dip == approx([-3.05009553, 1.77922239, -0.23558438])


if __name__ == "__main__":
    test_core_energy()
    test_dipole_rhf()
    test_dipole_uhf()
