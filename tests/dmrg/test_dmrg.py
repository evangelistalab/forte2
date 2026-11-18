import numpy as np
import pytest

from forte2 import System, RHF, State, CI, CISolver, DMRGSolver, MCOptimizer
from forte2.base_classes import DMRGParams
from forte2.dmrg import BLOCK2_AVAILABLE
from forte2.helpers.comparisons import approx, approx_abs

pytestmark = pytest.mark.skipif(not BLOCK2_AVAILABLE, reason="block2 is not installed")


def _dmrg_params(scratch, orbital_ordering="fiedler", twosite_to_onesite=None):
    # Six sites, so a bond dimension of a few hundred is exact
    return DMRGParams(
        bond_dims=[200] * 4 + [400] * 4,
        noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
        davidson_tols=[1e-12] * 8,
        n_sweeps=12,
        twosite_to_onesite=twosite_to_onesite,
        orbital_ordering=orbital_ordering,
        scratch=scratch,
    )


def test_dmrgscf_n2_vs_casscf(tmp_path):
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.1
    """
    state = State(nel=14, multiplicity=1, ms=0.0)
    core_orbitals = [0, 1, 2, 3]
    active_orbitals = [4, 5, 6, 7, 8, 9]

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        states=state, core_orbitals=core_orbitals, active_orbitals=active_orbitals
    )
    mc = MCOptimizer(ci_solver, final_orbitals="original")(rhf)
    mc.run()

    dmrg_solver = DMRGSolver(
        states=state,
        core_orbitals=core_orbitals,
        active_orbitals=active_orbitals,
        dmrg_params=_dmrg_params(str(tmp_path), orbital_ordering="gaopt"),
    )
    dmrgscf = MCOptimizer(dmrg_solver, final_orbitals="original")(rhf)
    dmrgscf.run()

    assert dmrg_solver.get_convergence_status() == [True]
    assert dmrgscf.E_ci[0] == approx(mc.E_ci[0])

    g1 = dmrg_solver.make_rdm(0, order=1, spin_type="sf")
    g2 = dmrg_solver.make_rdm(0, order=2, spin_type="sf")
    np.testing.assert_allclose(
        g1, ci_solver.make_rdm(0, order=1, spin_type="sf"), atol=1e-5
    )
    np.testing.assert_allclose(
        g2, ci_solver.make_rdm(0, order=2, spin_type="sf"), atol=1e-5
    )
    np.testing.assert_allclose(
        dmrg_solver.make_cumulant(0, order=2, spin_type="sf"),
        ci_solver.make_cumulant(0, order=2, spin_type="sf"),
        atol=1e-5,
    )

    ints = dmrg_solver.make_active_space_ints()
    e_rdm = (
        ints.E
        + np.einsum("pq,pq", ints.H, g1)
        + 0.5 * np.einsum("pqrs,pqrs", ints.V, g2)
    )
    assert e_rdm == approx_abs(mc.E_ci[0], 1e-6)


def test_dmrg_state_averaged_h2o_with_symmetry(tmp_path):
    # Active orbitals span the A1, B1, and B2 irreps of C2v
    xyz = """
    O 0.0  0.000  0.117
    H 0.0  0.757 -0.469
    H 0.0 -0.757 -0.469
    """
    states = [
        State(nel=10, multiplicity=1, ms=0.0, symmetry=0),
        State(nel=10, multiplicity=1, ms=0.0, symmetry=2),
        State(nel=10, multiplicity=3, ms=0.0, symmetry=3),
    ]
    nroots = [2, 1, 1]
    core_orbitals = [0]
    active_orbitals = [1, 2, 3, 4, 5, 6]

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        states=states,
        nroots=nroots,
        core_orbitals=core_orbitals,
        active_orbitals=active_orbitals,
    )
    ci = CI(ci_solver, final_orbitals="natural")(rhf)
    ci.run()

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    dmrg_solver = DMRGSolver(
        states=states,
        nroots=nroots,
        core_orbitals=core_orbitals,
        active_orbitals=active_orbitals,
        dmrg_params=_dmrg_params(str(tmp_path), twosite_to_onesite=8),
    )
    dmrg = CI(dmrg_solver, final_orbitals="natural")(rhf)
    dmrg.run()

    assert dmrg.E_ci == approx(ci.E_ci)
    assert dmrg.E_avg == approx(ci.E_avg)
    np.testing.assert_allclose(dmrg_solver.nat_occs, ci_solver.nat_occs, atol=1e-5)
    np.testing.assert_allclose(
        dmrg_solver.nat_occs_avg, ci_solver.nat_occs_avg, atol=1e-5
    )
