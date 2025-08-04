import forte2
from forte2.scf import RHF
from forte2.helpers.comparisons import approx


def test_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT", point_group="c2v"
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
test_rhf()

def test_rhf_zero_electron():
    xyz = """
    H           0.000000000000     0.000000000000     0.000000000000
    H           0.000000000000     0.000000000000     1.000000000000
    """
    system = forte2.System(
        xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT"
    )
    scf = RHF(charge=2)(system)
    scf.run()
    assert scf.E == approx(system.nuclear_repulsion)


def test_rhf_zero_virtuals():
    erhf = -126.604573431517
    xyz = "Ne 0 0 0"
    system = forte2.System(
        xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="def2-universal-JKFIT"
    )
    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)


def test_rhf_cholesky():
    erhf = -76.021769351262
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = forte2.System(
        xyz=xyz, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=1e-10
    )

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)
