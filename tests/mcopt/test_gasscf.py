import pytest

from forte2 import System, RHF, MCOptimizer, State, CISolver
from forte2.helpers.comparisons import approx


def test_gasscf_1():
    erhf = -76.05702512779526
    emcscf = -76.1156924702

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0)(system)

    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[3], gas_max=[6]),
        core_orbitals=2,
        active_orbitals=(3, 3),
    )
    mc = MCOptimizer(
        ci_solver,
        freeze_inter_gas_rots=True,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_gasscf_h2o_core():
    erhf = -76.05702512779526

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0)(system)

    ci_solver = CISolver(
        states=[
            State(nel=10, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[1]),
            State(nel=10, multiplicity=3, ms=1.0, gas_min=[0], gas_max=[1]),
        ],
        nroots=[1, 1],
        weights=[[1.0], [3.0]],
        core_orbitals=[1],
        active_orbitals=[[0], [2, 3, 4, 5]],
    )
    mc = MCOptimizer(
        ci_solver,
        freeze_inter_gas_rots=True,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E_ci[0] == approx(-56.3948123402)
    assert mc.E_ci[1] == approx(-56.4171164248)


def test_gasscf_2():
    erhf = -76.05702512779526
    emcscf = -76.078664204560

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(
        xyz=xyz, basis_set="cc-pVTZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-12)(system)

    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[3], gas_max=[6]),
        core_orbitals=[0, 1],
        active_orbitals=[[2, 3, 4], [5]],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-8,
        g_tol=1e-7,
        maxiter=100,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


@pytest.mark.slow
def test_gasscf_3():
    erhf = -40.21254161940163
    emcscf = -29.7051233787

    xyz = """
    C            0.052417904862     0.008091170764     0.039717608738
    H            1.170469902710     0.450548753521     1.737652542463
    H            1.167683656136     0.451231405150    -1.659924552573
    H           -1.690971530149     1.143552334703     0.040997211401
    H           -0.436869468885    -2.014192012525     0.039431679323
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVTZ",
        auxiliary_basis_set="def2-universal-jkfit",
        unit="bohr",
    )

    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-12)(system)

    ci_solver = CISolver(
        State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
        active_orbitals=[[0], [1, 2, 3, 4, 5, 6, 7, 8]],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-8,
        g_tol=1e-7,
        maxiter=500,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_gasscf_5():
    erhf = -76.02146209546578
    emcscf = -76.0776921745

    xyz = """
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-12)(system)

    ci_solver = CISolver(
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[6, 0], gas_max=[8, 2]),
        core_orbitals=[0],
        active_orbitals=[[1, 2, 3, 4], [5, 6]],
    )
    mc = MCOptimizer(
        ci_solver,
        freeze_inter_gas_rots=True,
        e_tol=1e-10,
        g_tol=1e-8,
    )(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)


def test_gasscf_transition_dipole():
    # Validated against the following forte v1 input:
    # import forte
    # molecule h2o{
    #     O            0.000000000000     0.000000000000    -0.069592187400
    #     H            0.000000000000    -0.783151105291     0.552239257834
    #     H            0.000000000000     0.783151105291     0.552239257834
    # symmetry c1
    # }

    # set global {
    #   scf_type                                df
    #   basis                                   cc-pvdz
    #   reference                               rhf
    #   df_basis_scf                            def2-universal-jkfit
    #   df_basis_mp2                            def2-universal-jkfit
    # }

    # set forte {
    #   active_space_solver                     genci
    #   int_type                                df
    #   avg_state                               [[0,1,1],[0,1,1]]
    #   avg_weight                              [[0],[1]]
    #   e_convergence                           12
    #   r_convergence                           8
    #   rotate_mos                              [1, 1,2]
    #   restricted_docc                         [1]
    #   gas1                                    [1]
    #   gas2                                    [5]
    #   gas1min                                 [2,1]
    #   gas1max                                 [2,1]
    #   transition_dipoles                      [[0,1,0]]
    #   mcscf_maxiter                           400
    #   mcscf_e_convergence                     1e-10
    #   mcscf_g_convergence                     1e-8
    # }

    # energy('forte')

    xyz = """
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, e_tol=1e-12, d_tol=1e-12)(system)

    ci_solver = CISolver(
        states=[
            State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
            State(nel=10, multiplicity=1, ms=0.0, gas_min=[2], gas_max=[2]),
        ],
        core_orbitals=[1],
        active_orbitals=[[0], [2, 3, 4, 5, 6]],
        nroots=[1, 1],
        weights=[[1.0], [0.0]],
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-10,
        g_tol=1e-8,
        do_transition_dipole=True,
    )(rhf)
    mc.run()

    assert ci_solver.E[0] == approx(-56.3204849516)
    assert ci_solver.E[1] == approx(-75.8022869454)

    assert ci_solver.oscillator_strengths[(1, 0)] == pytest.approx(
        0.00014521738399236842, abs=1e-5
    )
