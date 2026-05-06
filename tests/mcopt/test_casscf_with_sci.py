from forte2 import System, State, MCOptimizer
from forte2.scf import RHF
from forte2.sci import SelectedCISolver
from forte2.helpers.comparisons import approx
from forte2.base_classes.params import SelectedCIParams, DavidsonLiuParams


def test_sciscf_n2_multiple_roots():
    """Test that multiple roots can be converged for N2."""
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.1
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0)(system)
    ci_solver = SelectedCISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=4,
        active_orbitals=6,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-8,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            screening_criterion="hbci",
            guess_occ_window=2,
            guess_vir_window=2,
            num_threads=4,
            num_batches_per_thread=16,
        ),
        die_if_not_converged=False,
        nroots=2,
        davidson_liu_params=DavidsonLiuParams(
            e_tol=1e-10,
            r_tol=1e-5,
            ndets_per_guess=20,
        ),
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()
    assert ci_solver.E[0] == approx(-109.0799734286)
    assert ci_solver.E[1] == approx(-108.6858467105)
