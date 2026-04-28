from forte2 import System, RHF, MCOptimizer, AVAS, State, CISolver
from forte2.helpers.comparisons import approx


def test_sa_casscf_c2_transition_dipole():
    """Test CASSCF with C2 molecule."""
    xyz = """
    C 0.0 0.0 0.0
    C 0.0 0.0 1.2
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    singlet = State(nel=rhf.nel, multiplicity=1, ms=0.0)
    triplet = State(nel=rhf.nel, multiplicity=3, ms=0.0)
    avas = AVAS(
        selection_method="separate",
        num_active_docc=4,
        num_active_uocc=4,
        subspace=["C(2s)", "C(2p)"],
    )(rhf)
    ci_solver = CISolver([singlet, triplet], nroots=[2, 2])
    mc = MCOptimizer(ci_solver, do_transition_dipole=True)(avas)
    mc.run()
    assert mc.ci_solver.evals_per_solver[0][0] == approx(-75.6107427252)
    assert mc.ci_solver.evals_per_solver[0][1] == approx(-75.5314535451)
    assert mc.ci_solver.evals_per_solver[1][0] == approx(-75.5792187010)
    assert mc.ci_solver.evals_per_solver[1][1] == approx(-75.5789867708)
    assert mc.ci_solver.oscillator_strengths[(0, 1)] == approx(0.006525243082121279)
