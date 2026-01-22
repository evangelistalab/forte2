from forte2 import System, RHF, MCOptimizer, State
from forte2.helpers.comparisons import approx


def test_casscf_symmetry_breaking():
    """
    Test CASSCF with BeH2 molecule.
    The solution breaks the symmetry of the molecule due to the small active space.
    """

    erhf = -15.59967761106774
    emcscf = -15.6284019812

    xyz = """
    Be        0.000000000000     0.000000000000     0.000000000000
    H         0.000000000000     1.389990000000     2.500000000000
    H         0.000000000000    -1.390010000000     2.500000000000
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )

    rhf = RHF(charge=0, econv=1e-10)(system)
    mc = MCOptimizer(
        State(nel=6, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1],
        active_orbitals=[2, 3],
        econv=1e-9,
        maxiter=500,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)
