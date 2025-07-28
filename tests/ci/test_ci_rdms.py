from forte2 import *
from forte2.helpers.comparisons import approx


def compare_rdms(ci, root):
    import numpy as np

    rdm_threshold = 1e-12

    # Test the RDMs by computing them using the debug implementation
    ci_vec_det = np.zeros((ci.ndet))
    ci.spin_adapter.csf_C_to_det_C(ci.evecs[:, root], ci_vec_det)

    # Test the 1-RDMs
    a_1rdm = ci.ci_sigma_builder.a_1rdm(ci_vec_det, ci_vec_det, True)
    a_1rdm_debug = ci.ci_sigma_builder.a_1rdm_debug(ci_vec_det, ci_vec_det, True)
    assert (
        np.linalg.norm(a_1rdm - a_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between a_1rdm and a_1rdm_debug is too large: {np.linalg.norm(a_1rdm - a_1rdm_debug):.12f}."

    b_1rdm1 = ci.ci_sigma_builder.a_1rdm(ci_vec_det, ci_vec_det, False)
    b_1rdm1_debug = ci.ci_sigma_builder.a_1rdm_debug(ci_vec_det, ci_vec_det, False)
    assert (
        np.linalg.norm(b_1rdm1 - b_1rdm1_debug) < rdm_threshold
    ), f"Norm of the difference between b_1rdm1 and b_1rdm1_debug is too large: {np.linalg.norm(b_1rdm1 - b_1rdm1_debug):.12f}."

    rdm1_sf = ci.make_sf_1rdm(ci.evecs[:, root])
    rdm1_sf_debug = ci.ci_sigma_builder.sf_rdm1_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm1_sf - rdm1_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm1_sf and rdm1_sf_debug is too large: {np.linalg.norm(rdm1_sf - rdm1_sf_debug):.12f}."

    # Test the 2-RDMs
    rdm2_aa = ci.ci_sigma_builder.aa_2rdm(ci_vec_det, ci_vec_det, True)
    rdm2_aa_debug = ci.ci_sigma_builder.aa_2rdm_debug(ci_vec_det, ci_vec_det, True)
    assert (
        np.linalg.norm(rdm2_aa - rdm2_aa_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_aa and rdm2_aa_debug is too large: {np.linalg.norm(rdm2_aa - rdm2_aa_debug):.12f}."

    rdm2_bb = ci.ci_sigma_builder.aa_2rdm(ci_vec_det, ci_vec_det, False)
    rdm2_bb_debug = ci.ci_sigma_builder.aa_2rdm_debug(ci_vec_det, ci_vec_det, False)
    assert (
        np.linalg.norm(rdm2_bb - rdm2_bb_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_bb and rdm2_bb_debug is too large: {np.linalg.norm(rdm2_bb - rdm2_bb_debug):.12f}."

    rdm2_ab = ci.ci_sigma_builder.ab_2rdm(ci_vec_det, ci_vec_det)
    rdm2_ab_debug = ci.ci_sigma_builder.ab_2rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm2_ab - rdm2_ab_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_ab and rdm2_ab_debug is too large: {np.linalg.norm(rdm2_ab - rdm2_ab_debug):.12f}."

    rdm2_sf = ci.make_sf_2rdm(ci.evecs[:, root])
    rdm2_sf_debug = ci.ci_sigma_builder.sf_rdm2_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm2_sf - rdm2_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_sf and rdm2_sf_debug is too large: {np.linalg.norm(rdm2_sf - rdm2_sf_debug):.12f}."

    rdm3_aaa = ci.ci_sigma_builder.aaa_3rdm(ci_vec_det, ci_vec_det, True)
    rdm3_aaa_debug = ci.ci_sigma_builder.aaa_3rdm_debug(ci_vec_det, ci_vec_det, True)
    assert (
        np.linalg.norm(rdm3_aaa - rdm3_aaa_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_aaa and rdm3_aaa_debug is too large: {np.linalg.norm(rdm3_aaa - rdm3_aaa_debug):.12f}."

    rdm3_bbb = ci.ci_sigma_builder.aaa_3rdm(ci_vec_det, ci_vec_det, False)
    rdm3_bbb_debug = ci.ci_sigma_builder.aaa_3rdm_debug(ci_vec_det, ci_vec_det, False)
    assert (
        np.linalg.norm(rdm3_bbb - rdm3_bbb_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_bbb and rdm3_bbb_debug is too large: {np.linalg.norm(rdm3_bbb - rdm3_bbb_debug):.12f}."

    rdm3_aab = ci.ci_sigma_builder.aab_3rdm(ci_vec_det, ci_vec_det)
    rdm3_aab_debug = ci.ci_sigma_builder.aab_3rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm3_aab - rdm3_aab_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_aab and rdm3_aab_debug is too large: {np.linalg.norm(rdm3_aab - rdm3_aab_debug):.12f}."

    rdm3_abb = ci.ci_sigma_builder.abb_3rdm(ci_vec_det, ci_vec_det)
    rdm3_abb_debug = ci.ci_sigma_builder.abb_3rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm3_abb - rdm3_abb_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_abb and rdm3_abb_debug is too large: {np.linalg.norm(rdm3_abb - rdm3_abb_debug):.12f}."

    rdm3_sf = ci.make_sf_3rdm(ci.evecs[:, root])
    rdm3_sf_debug = ci.ci_sigma_builder.sf_rdm3_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm3_sf - rdm3_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_sf and rdm3_sf_debug is too large: {np.linalg.norm(rdm3_sf - rdm3_sf_debug):.12f}."


    # Test the spin-free cumulants
    rdm2_cumulant = ci.ci_sigma_builder.sf_2cumulant(ci_vec_det, ci_vec_det)
    rdm2_cumulant_debug = ci.ci_sigma_builder.sf_2cumulant_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm2_cumulant - rdm2_cumulant_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_cumulant and rdm2_cumulant_debug is too large: {np.linalg.norm(rdm2_cumulant - rdm2_cumulant_debug):.12f}."

    rdm3_cumulant = ci.ci_sigma_builder.sf_3cumulant(ci_vec_det, ci_vec_det)
    rdm3_cumulant_debug = ci.ci_sigma_builder.sf_3cumulant_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm3_cumulant - rdm3_cumulant_debug) < rdm_threshold
    ), f"Norm of the difference between rdm3_cumulant and rdm3_cumulant_debug is too large: {np.linalg.norm(rdm3_cumulant - rdm3_cumulant_debug):.12f}."    



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
        orbitals=[1, 2, 3, 4, 5, 6],
        core_orbitals=[0],
        state=State(nel=10, multiplicity=1, ms=0.0),
        nroot=1,
        do_test_rdms=True,
    )(rhf)
    ci.run()
    compare_rdms(ci, 0)

    assert ci.E[0] == approx(-100.019788438077)

test_ci_rdms_1()