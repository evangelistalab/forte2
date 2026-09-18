import numpy as np

from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.dsrg import DSRG_MRPT2
from forte2.helpers.comparisons import approx, approx_abs

# The forte references come from its spin-integrated DSRG-MRPT2, which drops the
# Fock coupling between GASes from H(0) just as forte2 does; see
# DSRGBase._fock_actv_0th. The tight g_tol below is load-bearing: the DSRG energy
# is not invariant to inter-GAS rotations, the softest GASSCF directions, so the
# default g_tol=1e-7 leaves ~5e-8 Eh of slop.


def test_gas_dsrg_mrpt2_vs_forte():
    """H2O core hole against forte: C1, DF, cc-pVDZ, s=1.0.

    GAS1 is the O 1s, capped at one electron. forte: gas1 [1], gas2 [6],
    gas1max [1], mcscf_active_frozen_orbital [0], correlation_solver dsrg-mrpt2.
    """
    escf_forte = -76.021492628198
    eref_forte = -56.307846585586
    edsrg_forte = -56.484543303744
    erelax_forte = -56.485926738908

    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 103.1
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-10)(system)
    rhf.run()

    ci_solver = CISolver(
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[1]),
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6]],
    )
    mc = MCOptimizer(
        ci_solver,
        active_frozen_orbitals=[0],
        maxiter=500,
        e_tol=1e-12,
        g_tol=1e-10,
    )(rhf)
    mc.run()

    assert rhf.E == approx(escf_forte)
    assert mc.E == approx(eref_forte)

    dsrg = DSRG_MRPT2(flow_param=1.0, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 2] == approx(eref_forte)
    assert dsrg.relax_energies[0, 0] == approx(edsrg_forte)
    assert dsrg.relax_energies[0, 1] == approx(erelax_forte)

    # Nonzero GAS1-GAS2 coupling is what makes this a real GAS test. gas_corr
    # slices index the correlated space, as dsrg.fock does.
    gas = dsrg.mo_space.gas_corr
    assert dsrg.nact == 7
    assert len(gas) == 2
    assert np.abs(dsrg.fock[gas[0], gas[1]]).max() == approx_abs(0.169039, 1e-6)
    for g in gas:
        block = dsrg.fock[g, g]
        assert np.abs(block - np.diag(np.diag(block))).max() < 1e-9


def test_sa_gas_dsrg_mrpt2_vs_forte():
    """State-averaged H2O against forte: A1 ground plus B1 core-excited state.

    Exercises per-state GAS windows, symmetry-resolved roots, state averaging and
    relaxation together. forte: avg_state [[0,1,1],[2,1,1]], gas1max [2,1],
    calc_type sa, correlation_solver dsrg-mrpt2; its gas1 [1,0,0,0] /
    gas2 [3,0,1,2] select MOs 0 and 1-6.
    """
    escf_forte = -76.021492628196
    eref_forte = -65.876345332565
    edsrg_forte = -66.089549102803
    erelax_forte = -66.101678862707
    eroots_forte = [-76.201761280602, -56.001596444812]

    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 103.1
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        symmetry=True,
    )
    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-10)(system)
    rhf.run()

    assert system.point_group == "C2V"

    ci_solver = CISolver(
        states=[
            State(nel=10, multiplicity=1, ms=0.0, symmetry=0, gas_min=[0], gas_max=[2]),
            State(nel=10, multiplicity=1, ms=0.0, symmetry=2, gas_min=[0], gas_max=[1]),
        ],
        nroots=[1, 1],
        weights=[[1.0], [1.0]],
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6]],
    )
    mc = MCOptimizer(
        ci_solver,
        active_frozen_orbitals=[0],
        maxiter=500,
        e_tol=1e-12,
        g_tol=1e-10,
    )(rhf)
    mc.run()

    assert rhf.E == approx(escf_forte)
    assert mc.E == approx(eref_forte)

    dsrg = DSRG_MRPT2(flow_param=1.0, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 2] == approx(eref_forte)
    assert dsrg.relax_energies[0, 0] == approx(edsrg_forte)
    assert dsrg.relax_energies[0, 1] == approx(erelax_forte)

    # Both roots, not just their average: a cancelling error would still move
    # the core-excitation energy.
    for eigval, ref in zip(dsrg.relax_eigvals, eroots_forte):
        assert eigval == approx(ref)


def test_gas_dsrg_mrpt2_reduces_to_cas():
    """A GAS partition that restricts nothing must reproduce the CAS result.

    GAS1 holds 0-4 electrons in two orbitals, so the determinant space is exactly
    CAS(6,6), and the two GASes span different irreps so their Fock coupling
    vanishes: an exact-identity check on the GAS bookkeeping. Both sides are run
    here because the values banked in test_dsrg_mrpt2.py used the default g_tol.
    """
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = RHF(charge=0)(system)
    rhf.run()

    def run(active_orbitals, state):
        # A fresh reference per run: relaxation calls ci_solver.set_ints(), so a
        # shared parent would hand the next DSRG Hbar as its bare Hamiltonian.
        ci_solver = CISolver(
            states=state,
            core_orbitals=[0, 1, 2, 3],
            active_orbitals=active_orbitals,
        )
        mc = MCOptimizer(ci_solver, maxiter=500, e_tol=1e-12, g_tol=1e-10)(rhf)
        mc.run()
        dsrg = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc)
        dsrg.run()
        return mc, dsrg

    mc_cas, cas = run(
        [[4, 5, 6, 7, 8, 9]],
        State(nel=14, multiplicity=1, ms=0.0),
    )
    mc_gas, gas = run(
        [[4, 5], [6, 7, 8, 9]],
        State(nel=14, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[4]),
    )

    assert mc_cas.mo_space.ngas == 1
    assert gas.mo_space.ngas == 2

    # The precondition that makes this an identity.
    gas_slices = gas.mo_space.gas_corr
    assert np.abs(gas.fock[gas_slices[0], gas_slices[1]]).max() < 1e-10

    assert mc_gas.E == approx_abs(mc_cas.E, 1e-10)
    assert gas.relax_energies[0, 0] == approx_abs(cas.relax_energies[0, 0], 1e-8)
    assert gas.relax_energies[0, 1] == approx_abs(cas.relax_energies[0, 1], 1e-8)
