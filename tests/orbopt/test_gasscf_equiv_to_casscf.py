from forte2 import System, RHF, MCOptimizer, State
from forte2.helpers.comparisons import approx


def test_gasscf_equiv_to_casscf():
    """With one GAS empty, this test should be equivalent to CASSCF."""
    erhf = -76.05702512779526
    ecasscf = -76.116395214672

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-12)(system)

    casscf = MCOptimizer(
        State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1],
        active_orbitals=[[2, 3, 4, 5, 6, 7], []],
        do_diis=False,
        econv=1e-8,
        gconv=1e-7,
        maxiter=500,
    )(rhf)
    casscf.run()

    assert rhf.E == approx(erhf)
    assert casscf.E == approx(ecasscf)
