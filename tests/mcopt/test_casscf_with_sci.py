import logging

import pytest

from forte2 import System, State, MCOptimizer
from forte2.scf import RHF
from forte2.sci import SelectedCISolver
from forte2.helpers.comparisons import approx
from forte2.base_classes.params import SelectedCIParams, DavidsonLiuParams

# Here we test two cases:
# 1. A tight threshold for the SCI solver. In this case, the final orbital
#    canonicalization does trigger a warning about orbital rotation invariance,
#    and the energies are correct.
# 2. A loose threshold for the SCI solver. In this case, the final orbital
#    canonicalization does trigger a warning about orbital rotation invariance,
#    and the energies are not correct.


def test_sciscf_can_reset_and_run_twice_with_same_solver():
    system = System(
        xyz="H 0 0 0\nH 0 0 1.4",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
        symmetry=False,
    )
    rhf = RHF(charge=0)(system)
    sci_params = SelectedCIParams(
        ci_algorithm="exact",
        guess_occ_window=1,
        guess_vir_window=1,
        var_threshold=1.0e-8,
        pt2_threshold=0.0,
    )
    ci_solver = SelectedCISolver(
        states=State(nel=2, multiplicity=1, ms=0.0),
        core_orbitals=0,
        active_orbitals=2,
        sci_params=sci_params,
    )
    mc = MCOptimizer(ci_solver, final_orbitals="original")(rhf)

    mc.run()
    first_energy = mc.E
    mc.reset()
    mc.run()

    assert mc.E == approx(first_energy)
    assert mc.E == approx(-1.137302245703818)
    assert mc.ci_solver is ci_solver
    assert ci_solver.sci_params is sci_params
    assert sci_params.guess_dets == []


@pytest.mark.parametrize(
    ("var_threshold", "expected_energies", "expect_rotation_warning"),
    [
        (1e-8, (-109.0799734286, -108.6858467105), False),
        (1e-2, None, True),
    ],
)
def test_sciscf_n2_multiple_roots(
    var_threshold, expected_energies, expect_rotation_warning, caplog
):
    """Test that multiple roots can be converged for N2 and warning about orbital rotation invariance."""
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
            var_threshold=var_threshold,
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
    assert not ci_solver.orbital_rotation_invariant

    mc = MCOptimizer(ci_solver)(rhf)
    with caplog.at_level(logging.CRITICAL):
        mc.run()

    rotation_warning = (
        "The active-space solver is not invariant to final orbital rotations"
    )
    assert (rotation_warning in caplog.text) is expect_rotation_warning
    if expected_energies is None:
        assert len(ci_solver.E) == 2
        assert ci_solver.E[0] < ci_solver.E[1]
    else:
        assert ci_solver.E[0] == approx(expected_energies[0])
        assert ci_solver.E[1] == approx(expected_energies[1])
