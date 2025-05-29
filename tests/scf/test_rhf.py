import forte2
from forte2.scf import RHF
import pytest

# assuming default scf tolerance of 1e-9
approx = lambda x: pytest.approx(x, rel=1e-8, abs=5e-8)


def test_rhf():
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


def test_rhf_zero_electron():
    xyz = """
    H           0.000000000000     0.000000000000     0.000000000000
    H           0.000000000000     0.000000000000     1.000000000000
    """
    system = forte2.System(xyz=xyz, basis="cc-pVQZ", auxiliary_basis="cc-pVQZ-JKFIT")
    scf = RHF(charge=2)(system)
    scf.run()
    assert scf.E == approx(system.nuclear_repulsion_energy())


def test_rhf_zero_virtuals():
    erhf = -126.604573431517
    xyz = "Ne 0 0 0"
    system = forte2.System(
        xyz=xyz, basis="sto-3g", auxiliary_basis="def2-universal-JKFIT"
    )
    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)


if __name__ == "__main__":
    test_rhf()
    test_rhf_zero_electron()
    test_rhf_zero_virtuals()
