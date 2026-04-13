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


def test_ci_tdm():
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
