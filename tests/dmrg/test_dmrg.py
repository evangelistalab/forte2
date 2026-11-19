import numpy as np
import pytest

from forte2 import System, RHF, State, CI, CISolver, DMRGSolver, MCOptimizer
from forte2.base_classes import DMRGParams
from forte2.dmrg import BLOCK2_AVAILABLE
from forte2.dsrg import DSRG_MRPT2
from forte2.helpers.comparisons import approx, approx_abs

pytestmark = pytest.mark.skipif(not BLOCK2_AVAILABLE, reason="block2 is not installed")


def _dmrg_params(scratch, orbital_ordering="fiedler", twosite_to_onesite=None):
    # Six sites, so a bond dimension of a few hundred is exact, and perturbative noise
    # can't move a solve off an excited state it lands on
    return DMRGParams(
        bond_dims=[200] * 4 + [400] * 4,
        noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
        noise_type="DensityMatrix",
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

    # DSRG-MRPT2 needs the 3-cumulant, and its relaxation restarts the DMRG
    dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()
    dmrg_dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(dmrgscf)
    dmrg_dsrg.run()
    np.testing.assert_allclose(
        dmrg_dsrg.relax_energies, dsrg.relax_energies, rtol=0, atol=5e-8
    )


def _flat(rdm):
    """Concatenate an RDM, or the blocks of a spin-dependent one, into one vector."""
    blocks = rdm if isinstance(rdm, tuple) else (rdm,)
    return np.concatenate([np.ravel(block) for block in blocks])


def test_dmrg_rdms_vs_ci(tmp_path):
    # The triplet at Ms = 1 has different alpha and beta RDMs. None of the roots are
    # degenerate, so transition RDMs between roots are defined up to their sign.
    xyz = """
    O 0.0  0.000  0.117
    H 0.0  0.757 -0.469
    H 0.0 -0.757 -0.469
    """
    states = [
        State(nel=10, multiplicity=1, ms=0.0),
        State(nel=10, multiplicity=3, ms=0.0),
        State(nel=10, multiplicity=3, ms=1.0),
    ]
    nroots = [3, 2, 1]
    core_orbitals = [0, 1]
    active_orbitals = [2, 3, 4, 5, 6, 7]

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(
        states=states,
        nroots=nroots,
        core_orbitals=core_orbitals,
        active_orbitals=active_orbitals,
    )
    ci = CI(ci_solver, final_orbitals="original")(rhf)
    ci.run()

    dmrg_solver = DMRGSolver(
        states=states,
        nroots=nroots,
        core_orbitals=core_orbitals,
        active_orbitals=active_orbitals,
        dmrg_params=_dmrg_params(str(tmp_path)),
    )
    dmrg = CI(dmrg_solver, final_orbitals="original")(rhf)
    dmrg.run()

    assert dmrg.E_ci == approx(ci.E_ci)
    # Pairs of absolute roots within the singlet and the Ms = 0 triplet
    for left, right in [(0, 0), (2, 2), (3, 3), (5, 5), (0, 1), (2, 0), (3, 4)]:
        sign = 1.0
        if left != right:
            # Both solvers fix each root only up to sign, so match the sign once and
            # apply it to every RDM of the pair
            ci_tdm = ci_solver.make_rdm(left, right, order=1, spin_type="sf")
            dmrg_tdm = dmrg_solver.make_rdm(left, right, order=1, spin_type="sf")
            largest = np.unravel_index(np.argmax(np.abs(ci_tdm)), ci_tdm.shape)
            sign = np.sign(ci_tdm[largest] * dmrg_tdm[largest])
        for spin_type, packed in [("sf", True), ("sd", True), ("sd", False)]:
            for order in (1, 2, 3):
                kwargs = dict(order=order, spin_type=spin_type, packed=packed)
                np.testing.assert_allclose(
                    sign * _flat(dmrg_solver.make_rdm(left, right, **kwargs)),
                    _flat(ci_solver.make_rdm(left, right, **kwargs)),
                    atol=1e-5,
                )
    for order in (2, 3):
        np.testing.assert_allclose(
            dmrg_solver.make_cumulant(5, order=order, spin_type="sf"),
            ci_solver.make_cumulant(5, order=order, spin_type="sf"),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            dmrg_solver.make_average_cumulant(order),
            ci_solver.make_average_cumulant(order),
            atol=1e-5,
        )


def test_dmrgscf_two_orbitals(tmp_path):
    # With two sites, the wavefunction center sits at both ends of the chain, and
    # one-site sweeps are skipped
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """
    state = State(nel=2, multiplicity=1, ms=0.0)

    system = System(xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pvtz-jkfit")
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci_solver = CISolver(states=state, active_orbitals=[0, 1])
    mc = MCOptimizer(ci_solver, final_orbitals="original")(rhf)
    mc.run()

    dmrg_solver = DMRGSolver(
        states=state,
        active_orbitals=[0, 1],
        dmrg_params=_dmrg_params(str(tmp_path), twosite_to_onesite=8),
    )
    dmrgscf = MCOptimizer(dmrg_solver, final_orbitals="original")(rhf)
    dmrgscf.run()

    assert dmrgscf.E_ci[0] == approx(mc.E_ci[0])


def test_dmrgscf_state_averaged_h2o_with_symmetry(tmp_path):
    # Active orbitals span the A1, B1, and B2 irreps of C2v. The DMRG-SCF iterations
    # restart from the previous, one-site MPS of each state, and the final rotation
    # to natural orbitals starts from a random MPS again.
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
    mc = MCOptimizer(ci_solver, final_orbitals="natural")(rhf)
    mc.run()

    dmrg_solver = DMRGSolver(
        states=states,
        nroots=nroots,
        core_orbitals=core_orbitals,
        active_orbitals=active_orbitals,
        dmrg_params=_dmrg_params(str(tmp_path), twosite_to_onesite=8),
    )
    dmrgscf = MCOptimizer(dmrg_solver, final_orbitals="natural")(rhf)
    dmrgscf.run()

    # Only the average energy is stationary in the orbitals, so individual roots of two
    # runs converged to the same tolerances can differ by about 1e-7
    assert dmrgscf.E_avg == approx(mc.E_avg)
    assert dmrgscf.E_ci == approx_abs(mc.E_ci, 1e-6)
    np.testing.assert_allclose(dmrg_solver.nat_occs, ci_solver.nat_occs, atol=1e-5)
    np.testing.assert_allclose(
        dmrg_solver.nat_occs_avg, ci_solver.nat_occs_avg, atol=1e-5
    )
