from forte2 import System, RHF, MCOptimizer, AVAS, State
from forte2.helpers.comparisons import approx


def test_casscf_cyclopropene():
    """Test CASSCF with cyclopropene (C3H4) molecule."""

    erhf = -114.40009162104958
    emcscf = -114.440831983407

    xyz = """
    H   0.912650   0.000000   1.457504
    H  -0.912650   0.000000   1.457504
    H   0.000000  -1.585659  -1.038624
    H   0.000000   1.585659  -1.038624
    C   0.000000   0.000000   0.859492
    C   0.000000  -0.651229  -0.499559
    C   0.000000   0.651229  -0.499559
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )

    rhf = RHF(charge=0, econv=1e-6)(system)
    avas = AVAS(
        subspace=["C(2p)"],
        subspace_pi_planes=[["C1-3"]],
        selection_method="total",
        num_active=3,
    )(rhf)
    mc = MCOptimizer(State(nel=rhf.nel, multiplicity=1, ms=0.0))(avas)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
