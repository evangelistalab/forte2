from forte2 import System
from forte2.scf import RHF
from forte2.helpers.comparisons import approx


def test_rhf():
    erhf = -76.061466407195
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)


def test_rhf_zero_electron():
    xyz = """
    H           0.000000000000     0.000000000000     0.000000000000
    H           0.000000000000     0.000000000000     1.000000000000
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    scf = RHF(charge=2)(system)
    scf.run()
    assert scf.E == approx(system.nuclear_repulsion)


def test_rhf_zero_virtuals():
    erhf = -126.604573431517
    xyz = "Ne 0 0 0"
    system = System(
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

    system = System(xyz=xyz, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=1e-10)

    scf = RHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(erhf)


def test_rhf_level_shift():
    # With DIIS, level shifting is usually not needed, but we provide
    # the functionality just in case.
    # With level_shift is 0.0, this system converges in around ~60 iterations,.
    # With level_shift=0.5, it converges in ~30 iterations.
    eref = -169.001135897278
    xyz = """
    C   0.000000  0.418626 0.000000
    H  -0.460595  1.426053 0.000000
    O   1.196516  0.242075 0.000000
    N  -0.936579 -0.568753 0.000000
    H  -0.634414 -1.530889 0.000000
    H  -1.921071 -0.362247 0.000000
    """
    system = System(xyz=xyz, basis_set="cc-pvtz", auxiliary_basis_set="cc-pvtz-jkfit")
    mf = RHF(charge=0, level_shift=0.5, do_diis=False)(system)
    mf.run()
    assert mf.E == approx(eref)
