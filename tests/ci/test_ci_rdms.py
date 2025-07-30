from forte2 import *
from forte2.helpers.comparisons import approx


def compare_rdms(ci, root):
    import numpy as np

    rdm_threshold = 1e-12

    ci_solver = ci.ci_solvers[0]

    # Test the 1-RDMs

    ci_vec_det = np.zeros((ci_solver.ndet))
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, root], ci_vec_det)

    a_1rdm, b_1rdm = ci_solver.make_sd_1rdm(root)

    a_1rdm_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_vec_det, ci_vec_det, True)
    assert (
        np.linalg.norm(a_1rdm - a_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between a_1rdm and a_1rdm_debug is too large: {np.linalg.norm(a_1rdm - a_1rdm_debug):.12f}."

    b_1rdm_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(
        ci_vec_det, ci_vec_det, False
    )
    assert (
        np.linalg.norm(b_1rdm - b_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between b_1rdm and b_1rdm_debug is too large: {np.linalg.norm(b_1rdm - b_1rdm_debug):.12f}."

    sf_1rdm = ci_solver.make_sf_1rdm(root)
    sf_1rdm_debug = ci_solver.ci_sigma_builder.sf_1rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(sf_1rdm - sf_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between sf_1rdm and sf_1rdm_debug is too large: {np.linalg.norm(sf_1rdm - sf_1rdm_debug):.12f}."


    # Test the 2-RDMs

    aa_2rdm, ab_2rdm, bb_2rdm = ci_solver.make_sd_2rdm(root, root)

    aa_2rdm_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(
        ci_vec_det, ci_vec_det, True
    )
    assert (
        np.linalg.norm(aa_2rdm - aa_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aa_2rdm and aa_2rdm_debug is too large: {np.linalg.norm(aa_2rdm - aa_2rdm_debug):.12f}."

    bb_2rdm_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(
        ci_vec_det, ci_vec_det, False
    )
    assert (
        np.linalg.norm(bb_2rdm - bb_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between bb_2rdm and bb_2rdm_debug is too large: {np.linalg.norm(bb_2rdm - bb_2rdm_debug):.12f}."

    ab_2rdm_debug = ci_solver.ci_sigma_builder.ab_2rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(ab_2rdm - ab_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between ab_2rdm and ab_2rdm_debug is too large: {np.linalg.norm(ab_2rdm - ab_2rdm_debug):.12f}."

    rdm2_sf = ci_solver.make_sf_2rdm(root)
    rdm2_sf_debug = ci_solver.ci_sigma_builder.sf_2rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(rdm2_sf - rdm2_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_sf and rdm2_sf_debug is too large: {np.linalg.norm(rdm2_sf - rdm2_sf_debug):.12f}."


    # Test the 3-RDMs

    aaa_3rdm, aab_3rdm, abb_3rdm, bbb_3rdm = ci_solver.make_sd_3rdm(root, root)

    aaa_3rdm_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_vec_det, ci_vec_det, True
    )
    assert (
        np.linalg.norm(aaa_3rdm - aaa_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aaa_3rdm and aaa_3rdm_debug is too large: {np.linalg.norm(aaa_3rdm - aaa_3rdm_debug):.12f}."

    bbb_3rdm_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_vec_det, ci_vec_det, False
    )
    assert (
        np.linalg.norm(bbb_3rdm - bbb_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between bbb_3rdm and bbb_3rdm_debug is too large: {np.linalg.norm(bbb_3rdm - bbb_3rdm_debug):.12f}."

    aab_3rdm_debug = ci_solver.ci_sigma_builder.aab_3rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(aab_3rdm - aab_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aab_3rdm and aab_3rdm_debug is too large: {np.linalg.norm(aab_3rdm - aab_3rdm_debug):.12f}."

    abb_3rdm_debug = ci_solver.ci_sigma_builder.abb_3rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(abb_3rdm - abb_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between abb_3rdm and abb_3rdm_debug is too large: {np.linalg.norm(abb_3rdm - abb_3rdm_debug):.12f}."

    sf_3rdm = ci_solver.make_sf_3rdm(root)
    sf_3rdm_debug = ci_solver.ci_sigma_builder.sf_3rdm_debug(ci_vec_det, ci_vec_det)
    assert (
        np.linalg.norm(sf_3rdm - sf_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between sf_3rdm and sf_3rdm_debug is too large: {np.linalg.norm(sf_3rdm - sf_3rdm_debug):.12f}."


    # Test the spin-free cumulants

    sf_2cumulant = ci_solver.make_sf_2cumulant(root)
    sf_2cumulant_debug = ci_solver.ci_sigma_builder.sf_2cumulant_debug(
        ci_vec_det, ci_vec_det
    )
    assert (
        np.linalg.norm(sf_2cumulant - sf_2cumulant_debug) < rdm_threshold
    ), f"Norm of the difference between sf_2cumulant and sf_2cumulant_debug is too large: {np.linalg.norm(sf_2cumulant - sf_2cumulant_debug):.12f}."

    sf_3cumulant = ci_solver.make_sf_3cumulant(root)
    sf_3cumulant_debug = ci_solver.ci_sigma_builder.sf_3cumulant_debug(
        ci_vec_det, ci_vec_det
    )
    assert (
        np.linalg.norm(sf_3cumulant - sf_3cumulant_debug) < rdm_threshold
    ), f"Norm of the difference between sf_3cumulant and sf_3cumulant_debug is too large: {np.linalg.norm(sf_3cumulant - sf_3cumulant_debug):.12f}."


def test_ci_rdms_1():
    xyz = """
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
