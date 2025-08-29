import pytest
from forte2 import System, RHF, CCSD, ROHF
from forte2.helpers.comparisons import approx
from forte2 import set_verbosity_level
# set_verbosity_level(5)

def test_ccsd_1():
    xyz = f"""
    C 0.0 0.0 0.0
    C 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", cholesky_tei=True, cholesky_tol=1.0e-010, point_group="D2H", reorient=True)

    rhf = RHF(charge=0, econv=1e-12)(system).run()

    cc = CCSD(rhf, frozen=4, econv=1e-12).run()

    assert cc.E == approx(-0.243926801139)

def test_ccsd_2():
    xyz = f"""
    F 0.0 0.0 -2.66816
    F 0.0 0.0 2.66816
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", cholesky_tei=True, cholesky_tol=1.0e-010, point_group="D2H", reorient=True, unit="bohr")

    rhf = ROHF(charge=0, econv=1e-12, ms=1)(system).run()

    cc = CCSD(rhf, frozen=4, econv=1e-12).run()

    assert cc.E == approx(-0.53380816)


@pytest.mark.skip(reason="Test is too large to be run")
def test_ccsd_3():
    xyz = """
        O -2.877091949897 -1.507375565672 -0.398996049903
        C -0.999392972049 -0.222326510867 0.093940021615
        C 1.633098051399 -1.126399113321 0.723677865007
        O -1.316707936360 2.330484008081 0.195537896270
        N 3.588772131647 0.190046035276 -0.635572324857
        H 1.738434758147 -3.192291478262 0.201142047999
        H 1.805107822402 -0.972547254301 2.850386782716
        H 3.367427816470 2.065392438845 -0.521139962778
        H 5.288732713155 -0.301105855518 0.028508872837
        H -3.050135067115 2.755707159769 -0.234244183166
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", unit="bohr", auxiliary_basis_set="cc-pvqz-jkfit")

    rhf = RHF(charge=0, econv=1.0e-010)(system).run()

    cc = CCSD(rhf, frozen=10, econv=1.0e-010).run()
