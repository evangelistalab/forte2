import numpy as np

from forte2 import (
    System,
    RHF,
    CI,
    State,
    compute_a_1rdm,
    compute_b_1rdm,
    compute_aa_2rdm,
    compute_ab_2rdm,
    compute_bb_2rdm,
    cpp_helpers,
)
from forte2.helpers.comparisons import approx
from forte2.base_classes import DavidsonLiuParams


def test_ci_tdm_same_solver():
    xyz = """
    N 0.0 0.0 -1.0
    N 0.0 0.0 1.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=[0, 1, 2, 3],
        active_orbitals=[4, 5, 6, 7, 8, 9],
        nroots=3,
    )(rhf)
    ci.run()

    for root_left in range(3):
        for root_right in range(3):
            c_l = ci.sub_solvers[0].csf_C_to_det_C(
                ci.sub_solvers[0].evecs[:, root_left]
            )
            state_left = ci.sub_solvers[0].ci_sigma_builder.make_sparse_state(c_l)

            c_r = ci.sub_solvers[0].csf_C_to_det_C(
                ci.sub_solvers[0].evecs[:, root_right]
            )
            state_right = ci.sub_solvers[0].ci_sigma_builder.make_sparse_state(c_r)

            tdm1_a_ref = compute_a_1rdm(state_left, state_right, 6)
            tdm1_b_ref = compute_b_1rdm(state_left, state_right, 6)

            tdm1_a, tdm1_b = ci.sub_solvers[0].make_sd_1rdm(root_left, root_right)
            assert np.allclose(tdm1_a, tdm1_a_ref)
            assert np.allclose(tdm1_b, tdm1_b_ref)

            tdm2_aa_ref = compute_aa_2rdm(state_left, state_right, 6)
            tdm2_ab_ref = compute_ab_2rdm(state_left, state_right, 6)
            tdm2_bb_ref = compute_bb_2rdm(state_left, state_right, 6)

            tdm2_aa, tdm2_ab, tdm2_bb = ci.sub_solvers[0].make_sd_2rdm(
                root_left, root_right
            )
            tdm2_aa = cpp_helpers.packed_tensor4_to_tensor4(tdm2_aa)
            tdm2_bb = cpp_helpers.packed_tensor4_to_tensor4(tdm2_bb)
            assert np.allclose(tdm2_aa, tdm2_aa_ref)
            assert np.allclose(tdm2_ab, tdm2_ab_ref)
            assert np.allclose(tdm2_bb, tdm2_bb_ref)


def test_gasci_tdm():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    ci = CI(
        active_orbitals=[[0], [2, 3, 4, 5, 6]],
        states=[
            State(nel=10, multiplicity=1, ms=0.0, gas_min=[1], gas_max=[1]),
            State(nel=10, multiplicity=1, ms=0.0, gas_min=[2], gas_max=[2]),
        ],
        nroots=[1, 1],
    )(rhf)
    ci.run()

    left_solver = ci.sub_solvers[0]
    right_solver = ci.sub_solvers[1]
    left_sb = left_solver.ci_sigma_builder
    right_sb = right_solver.ci_sigma_builder

    C_left = left_solver.csf_C_to_det_C(left_solver.evecs[:, 0])
    C_right = right_solver.csf_C_to_det_C(right_solver.evecs[:, 0])

    state_left = left_solver.ci_sigma_builder.make_sparse_state(C_left)
    state_right = right_solver.ci_sigma_builder.make_sparse_state(C_right)

    a_1trdm = left_sb.a_1trdm(right_sb, C_left, C_right)
    a_1trdm_ref = compute_a_1rdm(state_left, state_right, 6)

    assert np.allclose(a_1trdm, a_1trdm_ref)
