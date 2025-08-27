from forte2 import System, RHF, CCSD, ROHF
from forte2.helpers.comparisons import approx


def test_ccsd_1():
    xyz = f"""
    C 0.0 0.0 0.0
    C 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", cholesky_tei=True, cholesky_tol=1.0e-010)

    rhf = RHF(charge=0, econv=1e-12)(system).run()

    cc = CCSD(rhf, frozen=4, econv=1e-12).run()

    assert cc.E == approx(-0.243926801139)
