from forte2 import System, GHF, RelMCOptimizer, State
from forte2.dsrg import DSRG_MRPT2
from forte2.helpers.comparisons import approx

def test_mrpt2_hf():
    erhf = -99.9977252002946
    emcscf = -100.0435018956

    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = GHF(charge=0, econv=1e-12)(system)
    mc = RelMCOptimizer(
        nel=10,
        active_orbitals=12,
        core_orbitals=2,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = DSRG_MRPT2()(mc)
    dsrg.run()
    print(dsrg.E)
test_mrpt2_hf()