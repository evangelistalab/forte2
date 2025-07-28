from forte2 import *
from forte2.helpers.comparisons import approx


def compare_rdms(ci, root):
    import numpy as np

    rdm_threshold = 1e-12

    ci_solver = ci.ci_solvers[0]

    # Test the RDMs by computing them using the debug implementation
    ci_vec_det = np.zeros((ci_solver.ndet))
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, root], ci_vec_det)

    rdm1_a = ci_solver.ci_sigma_builder.rdm1_a(ci_vec_det, ci_vec_det, True)
    rdm1_a_debug = ci_solver.ci_sigma_builder.rdm1_a_debug(ci_vec_det, ci_vec_det, True)
    assert (
        np.linalg.norm(rdm1_a - rdm1_a_debug) < rdm_threshold
    ), f"Norm of the difference between rdm1_a and rdm1_a_debug is too large: {np.linalg.norm(rdm1_a - rdm1_a_debug):.12f}."

    rdm1_b = ci_solver.ci_sigma_builder.rdm1_a(ci_vec_det, ci_vec_det, False)
    rdm1_b_debug = ci_solver.ci_sigma_builder.rdm1_a_debug(
        ci_vec_det, ci_vec_det, False
    )
    assert (
        np.linalg.norm(rdm1_b - rdm1_b_debug) < rdm_threshold
    ), f"Norm of the difference between rdm1_b and rdm1_b_debug is too large: {np.linalg.norm(rdm1_b - rdm1_b_debug):.12f}."

    rdm1_sf = ci_solver.make_rdm1_sf(ci_solver.evecs[:, root])
    rdm1_sf_debug = ci_solver.ci_sigma_builder.rdm1_sf_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm1_sf - rdm1_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm1_sf and rdm1_sf_debug is too large: {np.linalg.norm(rdm1_sf - rdm1_sf_debug):.12f}."

    rdm2_aa = ci_solver.ci_sigma_builder.rdm2_aa(ci_vec_det, ci_vec_det, True)
    rdm2_aa_debug = ci_solver.ci_sigma_builder.rdm2_aa_debug(
        ci_vec_det, ci_vec_det, True
    )
    assert (
        np.linalg.norm(rdm2_aa - rdm2_aa_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_aa and rdm2_aa_debug is too large: {np.linalg.norm(rdm2_aa - rdm2_aa_debug):.12f}."

    rdm2_bb = ci_solver.ci_sigma_builder.rdm2_aa(ci_vec_det, ci_vec_det, False)
    rdm2_bb_debug = ci_solver.ci_sigma_builder.rdm2_aa_debug(
        ci_vec_det, ci_vec_det, False
    )
    assert (
        np.linalg.norm(rdm2_bb - rdm2_bb_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_bb and rdm2_bb_debug is too large: {np.linalg.norm(rdm2_bb - rdm2_bb_debug):.12f}."

    rdm2_ab = ci_solver.ci_sigma_builder.rdm2_ab(ci_vec_det, ci_vec_det)
    rdm2_ab_debug = ci_solver.ci_sigma_builder.rdm2_ab_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm2_ab - rdm2_ab_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_ab and rdm2_ab_debug is too large: {np.linalg.norm(rdm2_ab - rdm2_ab_debug):.12f}."

    rdm2_sf = ci_solver.make_rdm2_sf(ci_solver.evecs[:, root])
    rdm2_sf_debug = ci_solver.ci_sigma_builder.rdm2_sf_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm2_sf - rdm2_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_sf and rdm2_sf_debug is too large: {np.linalg.norm(rdm2_sf - rdm2_sf_debug):.12f}."

    rdm3_aaa = ci_solver.ci_sigma_builder.rdm3_aaa(ci_vec_det, ci_vec_det, True)
    rdm3_aaa_debug = ci_solver.ci_sigma_builder.rdm3_aaa_debug(
        ci_vec_det, ci_vec_det, True
    )
    assert (
        np.linalg.norm(rdm3_aaa - rdm3_aaa_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_aaa and rdm3_aaa_debug is too large: {np.linalg.norm(rdm3_aaa - rdm3_aaa_debug):.12f}."

    rdm3_bbb = ci_solver.ci_sigma_builder.rdm3_aaa(ci_vec_det, ci_vec_det, False)
    rdm3_bbb_debug = ci_solver.ci_sigma_builder.rdm3_aaa_debug(
        ci_vec_det, ci_vec_det, False
    )
    assert (
        np.linalg.norm(rdm3_bbb - rdm3_bbb_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_bbb and rdm3_bbb_debug is too large: {np.linalg.norm(rdm3_bbb - rdm3_bbb_debug):.12f}."

    rdm3_aab = ci_solver.ci_sigma_builder.rdm3_aab(ci_vec_det, ci_vec_det)
    rdm3_aab_debug = ci_solver.ci_sigma_builder.rdm3_aab_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm3_aab - rdm3_aab_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_aab and rdm3_aab_debug is too large: {np.linalg.norm(rdm3_aab - rdm3_aab_debug):.12f}."

    rdm3_abb = ci_solver.ci_sigma_builder.rdm3_abb(ci_vec_det, ci_vec_det)
    rdm3_abb_debug = ci_solver.ci_sigma_builder.rdm3_abb_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm3_abb - rdm3_abb_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_abb and rdm3_abb_debug is too large: {np.linalg.norm(rdm3_abb - rdm3_abb_debug):.12f}."


def test_ci_rdms_1():
    xyz = f"""
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        do_test_rdms=True,
    )(rhf)
    ci.run()
    compare_rdms(ci, 0)

    assert ci.E[0] == approx(-100.019788438077)
