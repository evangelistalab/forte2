from forte2 import System, RHF, MCOptimizer, State
from forte2.helpers.comparisons import approx


def test_casscf_n2_cholesky():
    """Equivalent to casscf_8 test in Forte"""
    erhf = -108.949591958787
    emcscf = -109.090719613072

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.120
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        cholesky_tei=True,
        cholesky_tol=1e-10,
        symmetry=True,
    )
    rhf = RHF(charge=0, econv=1e-12)(system)

    mc = MCOptimizer(
        State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        econv=1e-9,
    )(rhf)
    mc.run()
    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
