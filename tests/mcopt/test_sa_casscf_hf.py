import pytest

from forte2 import System, RHF, MCOptimizer, State, CISolver
from forte2.helpers.comparisons import approx


def test_sa_casscf_hf():
    erhf = -100.009873562527
    emcscf_root_1 = -99.9964137656
    emcscf_root_2 = -99.6886809114
    emcscf_root_3 = -99.6886809114
    emcscf_root_4 = -99.4702123772
    emcscf_avg = -99.7109969914

    xyz = """
    H            0.000000000000     0.000000000000    -0.949624435830
    F            0.000000000000     0.000000000000     0.050375564170
    """

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5],
        nroots=4,
    )
    mc = MCOptimizer(
        ci_solver,
        do_transition_dipole=True,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf_avg)
    assert mc.E_ci[0] == approx(emcscf_root_1)
    assert mc.E_ci[1] == approx(emcscf_root_2)
    assert mc.E_ci[2] == approx(emcscf_root_3)
    assert mc.E_ci[3] == approx(emcscf_root_4)

    # if energy is converged to ~1e-8, properties are converged to ~1e-4
    # transition dipole moment of 0->3 transition
    assert abs(mc.ci_solver.tdm_per_solver[0][(0, 3)][2]) == pytest.approx(
        1.1357911621720147, abs=1e-4
    )
    # transition oscillator strength of 0->3 transition
    assert mc.ci_solver.fosc_per_solver[0][(0, 3)] == pytest.approx(
        0.4525407586932925, abs=1e-4
    )
    # total dipole of state 0, 1
    assert abs(mc.ci_solver.tdm_per_solver[0][(0, 0)][2]) == pytest.approx(
        0.7244112903260456, abs=1e-4
    )
    assert abs(mc.ci_solver.tdm_per_solver[0][(1, 1)][2]) == pytest.approx(
        0.8239033452222257, abs=1e-4
    )
