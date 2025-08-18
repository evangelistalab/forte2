from forte2 import System, RHF, MCOptimizer, State
from forte2.helpers.comparisons import approx


def test_casscf_singlet_benzyne():
    erhf = -226.40943786499565
    emcscf = -226.575743550979

    xyz = """
    C   0.0000000000  -2.5451795941   0.0000000000
    C   0.0000000000   2.5451795941   0.0000000000
    C  -2.2828001669  -1.3508352528   0.0000000000
    C   2.2828001669  -1.3508352528   0.0000000000
    C   2.2828001669   1.3508352528   0.0000000000
    C  -2.2828001669   1.3508352528   0.0000000000
    H  -4.0782187459  -2.3208602146   0.0000000000
    H   4.0782187459  -2.3208602146   0.0000000000
    H   4.0782187459   2.3208602146   0.0000000000
    H  -4.0782187459   2.3208602146   0.0000000000
    """

    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    mc = MCOptimizer(
        State(nel=40, multiplicity=1, ms=0.0),
        core_orbitals=list(range(19)),
        active_orbitals=[19, 20],
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
