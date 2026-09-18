from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.dsrg import DSRG_MRPT3
from forte2.helpers.comparisons import approx

# Same two systems and forte settings as test_gas_dsrg_mrpt2.py, with
# corr_level pt3. See that file for why the reference must be converged tightly
# and why the spin-adapted forte solver is the one to compare against.


def test_gas_dsrg_mrpt3_vs_forte():
    """H2O core hole against forte: C1, DF, cc-pVDZ, s=1.0.

    GAS1 is the O 1s, capped at one electron. forte: gas1 [1], gas2 [6],
    gas1max [1], mcscf_active_frozen_orbital [0], corr_level pt3.
    """
    eref_forte = -56.307846585586
    edsrg_forte = -56.492749407999
    erelax_forte = -56.493942820733

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

    assert mc.E == approx(eref_forte)

    dsrg = DSRG_MRPT3(flow_param=1.0, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 2] == approx(eref_forte)
    assert dsrg.relax_energies[0, 0] == approx(edsrg_forte)
    assert dsrg.relax_energies[0, 1] == approx(erelax_forte)


def test_sa_gas_dsrg_mrpt3_vs_forte():
    """State-averaged H2O against forte: A1 ground plus B1 core-excited state.

    forte: avg_state [[0,1,1],[2,1,1]], gas1max [2,1], calc_type sa,
    corr_level pt3.
    """
    eref_forte = -65.876345332565
    edsrg_forte = -66.083801169853
    erelax_forte = -66.093531257328
    eroots_forte = [-76.223673464995, -55.963389049660]

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

    assert mc.E == approx(eref_forte)

    dsrg = DSRG_MRPT3(flow_param=1.0, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, 2] == approx(eref_forte)
    assert dsrg.relax_energies[0, 0] == approx(edsrg_forte)
    assert dsrg.relax_energies[0, 1] == approx(erelax_forte)

    for eigval, ref in zip(dsrg.relax_eigvals, eroots_forte):
        assert eigval == approx(ref)
