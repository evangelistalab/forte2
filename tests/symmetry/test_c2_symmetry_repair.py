from forte2 import System, RHF, MCOptimizer, State
from forte2.helpers.comparisons import approx


def test_c2_symmetry_repair():
    """
    PySCF input:
    mol = pyscf.gto.M(atom=f"C ; C 1 1.800", basis="cc-pvdz", symmetry='d2h', spin=0, charge=0)
    mf = pyscf.scf.RHF(mol).density_fit("cc-pvtz-jkfit")
    mf.kernel()

    # mf.mol.build(0, 0, symmetry='d2h')
    mc = pyscf.mcscf.CASSCF(mf, 8, 8).state_average_(np.ones(3)/3)
    mc.fcisolver.conv_tol = 1e-9
    mc.fcisolver.wfnsym = 'Ag'

    res = mc.mc2step()
    """
    xyz = """
    C 0.0 0.0 0.0
    C 0.0 0.0 1.8
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )

    rhf = RHF(charge=0)(system)
    mcscf = MCOptimizer(
        states=State(nel=12, multiplicity=1, ms=0.0, symmetry=0),
        nroots=3,
        core_orbitals=2,
        active_orbitals=8,
    )(rhf)
    mcscf.run()

    ref_ci = [-75.4911788852, -75.4879859747, -75.4732134962]
    assert mcscf.E_ci == approx(ref_ci)
