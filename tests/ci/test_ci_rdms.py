import numpy as np

from forte2 import System, RHF, CI, State
from forte2.helpers.comparisons import approx


def compare_rdms(ci):
    rdm_threshold = 1e-12

    ci_solver = ci.sub_solvers[0]

    # Test the 1-RDMs

    ci_0_det = np.zeros((ci_solver.ndet))
    ci_1_det = np.zeros((ci_solver.ndet))
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 0], ci_0_det)
    ci_solver.spin_adapter.csf_C_to_det_C(ci_solver.evecs[:, 1], ci_1_det)

    a_1rdm_0, b_1rdm_0 = ci_solver.make_sd_1rdm(0)
    a_1rdm_1, b_1rdm_1 = ci_solver.make_sd_1rdm(1)

    a_1rdm_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_0_det, ci_0_det, True)
    assert (
        np.linalg.norm(a_1rdm_0 - a_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between a_1rdm_0 and a_1rdm_debug is too large: {np.linalg.norm(a_1rdm_0 - a_1rdm_debug):.12f}."

    b_1rdm_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_0_det, ci_0_det, False)
    assert (
        np.linalg.norm(b_1rdm_0 - b_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between b_1rdm_0 and b_1rdm_debug is too large: {np.linalg.norm(b_1rdm_0 - b_1rdm_debug):.12f}."

    sf_1rdm_0 = ci_solver.make_sf_1rdm(0)
    sf_1rdm_debug = ci_solver.ci_sigma_builder.sf_1rdm_debug(ci_0_det, ci_0_det)
    assert (
        np.linalg.norm(sf_1rdm_0 - sf_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between sf_1rdm_0 and sf_1rdm_debug is too large: {np.linalg.norm(sf_1rdm_0 - sf_1rdm_debug):.12f}."

    a_1rdm_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_1_det, ci_1_det, True)
    assert (
        np.linalg.norm(a_1rdm_1 - a_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between a_1rdm_1 and a_1rdm_debug is too large: {np.linalg.norm(a_1rdm_1 - a_1rdm_debug):.12f}."

    b_1rdm_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_1_det, ci_1_det, False)
    assert (
        np.linalg.norm(b_1rdm_1 - b_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between b_1rdm_1 and b_1rdm_debug is too large: {np.linalg.norm(b_1rdm_1 - b_1rdm_debug):.12f}."

    sf_1rdm_1 = ci_solver.make_sf_1rdm(1)
    sf_1rdm_debug = ci_solver.ci_sigma_builder.sf_1rdm_debug(ci_1_det, ci_1_det)
    assert (
        np.linalg.norm(sf_1rdm_1 - sf_1rdm_debug) < rdm_threshold
    ), f"Norm of the difference between sf_1rdm_1 and sf_1rdm_debug is too large: {np.linalg.norm(sf_1rdm_1 - sf_1rdm_debug):.12f}."

    # Test the 1-TDMs
    a_1tdm_01, b_1tdm_01 = ci_solver.make_sd_1rdm(0, 1)
    a_1tdm_10, b_1tdm_10 = ci_solver.make_sd_1rdm(1, 0)
    a_1tdm_01_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_0_det, ci_1_det, True)
    assert (
        np.linalg.norm(a_1tdm_01 - a_1tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between a_1tdm_01 and a_1tdm_01_debug is too large: {np.linalg.norm(a_1tdm_01 - a_1tdm_01_debug):.12f}."
    b_1tdm_01_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_0_det, ci_1_det, False)
    assert (
        np.linalg.norm(b_1tdm_01 - b_1tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between b_1tdm_01 and b_1tdm_01_debug is too large: {np.linalg.norm(b_1tdm_01 - b_1tdm_01_debug):.12f}."
    a_1tdm_10_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_1_det, ci_0_det, True)
    assert (
        np.linalg.norm(a_1tdm_10 - a_1tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between a_1tdm_10 and a_1tdm_10_debug is too large: {np.linalg.norm(a_1tdm_10 - a_1tdm_10_debug):.12f}."
    b_1tdm_10_debug = ci_solver.ci_sigma_builder.a_1rdm_debug(ci_1_det, ci_0_det, False)
    assert (
        np.linalg.norm(b_1tdm_10 - b_1tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between b_1tdm_10 and b_1tdm_10_debug is too large: {np.linalg.norm(b_1tdm_10 - b_1tdm_10_debug):.12f}."
    sf_1tdm_01 = ci_solver.make_sf_1rdm(0, 1)
    sf_1tdm_01_debug = ci_solver.ci_sigma_builder.sf_1rdm_debug(ci_0_det, ci_1_det)
    assert (
        np.linalg.norm(sf_1tdm_01 - sf_1tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between sf_1tdm_01 and sf_1tdm_01_debug is too large: {np.linalg.norm(sf_1tdm_01 - sf_1tdm_01_debug):.12f}."
    sf_1tdm_10 = ci_solver.make_sf_1rdm(1, 0)
    sf_1tdm_10_debug = ci_solver.ci_sigma_builder.sf_1rdm_debug(ci_1_det, ci_0_det)
    assert (
        np.linalg.norm(sf_1tdm_10 - sf_1tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between sf_1tdm_10 and sf_1tdm_10_debug is too large: {np.linalg.norm(sf_1tdm_10 - sf_1tdm_10_debug):.12f}."

    # Test the 2-RDMs

    aa_2rdm_0, ab_2rdm_0, bb_2rdm_0 = ci_solver.make_sd_2rdm(0)
    aa_2rdm_1, ab_2rdm_1, bb_2rdm_1 = ci_solver.make_sd_2rdm(1)

    aa_2rdm_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(ci_0_det, ci_0_det, True)
    assert (
        np.linalg.norm(aa_2rdm_0 - aa_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aa_2rdm and aa_2rdm_debug is too large: {np.linalg.norm(aa_2rdm_0 - aa_2rdm_debug):.12f}."

    bb_2rdm_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(ci_0_det, ci_0_det, False)
    assert (
        np.linalg.norm(bb_2rdm_0 - bb_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between bb_2rdm and bb_2rdm_debug is too large: {np.linalg.norm(bb_2rdm_0 - bb_2rdm_debug):.12f}."

    ab_2rdm_debug = ci_solver.ci_sigma_builder.ab_2rdm_debug(ci_0_det, ci_0_det)
    assert (
        np.linalg.norm(ab_2rdm_0 - ab_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between ab_2rdm and ab_2rdm_debug is too large: {np.linalg.norm(ab_2rdm_0 - ab_2rdm_debug):.12f}."

    rdm2_sf = ci_solver.make_sf_2rdm(0)
    rdm2_sf_debug = ci_solver.ci_sigma_builder.sf_2rdm_debug(ci_0_det, ci_0_det)
    assert (
        np.linalg.norm(rdm2_sf - rdm2_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_sf and rdm2_sf_debug is too large: {np.linalg.norm(rdm2_sf - rdm2_sf_debug):.12f}."

    aa_2rdm_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(ci_1_det, ci_1_det, True)
    assert (
        np.linalg.norm(aa_2rdm_1 - aa_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aa_2rdm and aa_2rdm_debug is too large: {np.linalg.norm(aa_2rdm_1 - aa_2rdm_debug):.12f}."

    bb_2rdm_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(ci_1_det, ci_1_det, False)
    assert (
        np.linalg.norm(bb_2rdm_1 - bb_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between bb_2rdm and bb_2rdm_debug is too large: {np.linalg.norm(bb_2rdm_1 - bb_2rdm_debug):.12f}."

    ab_2rdm_debug = ci_solver.ci_sigma_builder.ab_2rdm_debug(ci_1_det, ci_1_det)
    assert (
        np.linalg.norm(ab_2rdm_1 - ab_2rdm_debug) < rdm_threshold
    ), f"Norm of the difference between ab_2rdm and ab_2rdm_debug is too large: {np.linalg.norm(ab_2rdm_1 - ab_2rdm_debug):.12f}."

    rdm2_sf = ci_solver.make_sf_2rdm(1)
    rdm2_sf_debug = ci_solver.ci_sigma_builder.sf_2rdm_debug(ci_1_det, ci_1_det)
    assert (
        np.linalg.norm(rdm2_sf - rdm2_sf_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_sf and rdm2_sf_debug is too large: {np.linalg.norm(rdm2_sf - rdm2_sf_debug):.12f}."

    # Test the 2-TDMs
    aa_2tdm_01, ab_2tdm_01, bb_2tdm_01 = ci_solver.make_sd_2rdm(0, 1)
    aa_2tdm_10, ab_2tdm_10, bb_2tdm_10 = ci_solver.make_sd_2rdm(1, 0)
    aa_2tdm_01_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(
        ci_0_det, ci_1_det, True
    )
    assert (
        np.linalg.norm(aa_2tdm_01 - aa_2tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between aa_2tdm_01 and aa_2tdm_01_debug is too large: {np.linalg.norm(aa_2tdm_01 - aa_2tdm_01_debug):.12f}."

    bb_2tdm_01_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(
        ci_0_det, ci_1_det, False
    )
    assert (
        np.linalg.norm(bb_2tdm_01 - bb_2tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between bb_2tdm_01 and bb_2tdm_01_debug is too large: {np.linalg.norm(bb_2tdm_01 - bb_2tdm_01_debug):.12f}."
    ab_2tdm_01_debug = ci_solver.ci_sigma_builder.ab_2rdm_debug(ci_0_det, ci_1_det)
    assert (
        np.linalg.norm(ab_2tdm_01 - ab_2tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between ab_2tdm_01 and ab_2tdm_01_debug is too large: {np.linalg.norm(ab_2tdm_01 - ab_2tdm_01_debug):.12f}."
    aa_2tdm_10_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(
        ci_1_det, ci_0_det, True
    )
    assert (
        np.linalg.norm(aa_2tdm_10 - aa_2tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between aa_2tdm_10 and aa_2tdm_10_debug is too large: {np.linalg.norm(aa_2tdm_10 - aa_2tdm_10_debug):.12f}."
    bb_2tdm_10_debug = ci_solver.ci_sigma_builder.aa_2rdm_debug(
        ci_1_det, ci_0_det, False
    )
    assert (
        np.linalg.norm(bb_2tdm_10 - bb_2tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between bb_2tdm_10 and bb_2tdm_10_debug is too large: {np.linalg.norm(bb_2tdm_10 - bb_2tdm_10_debug):.12f}."
    ab_2tdm_10_debug = ci_solver.ci_sigma_builder.ab_2rdm_debug(ci_1_det, ci_0_det)
    assert (
        np.linalg.norm(ab_2tdm_10 - ab_2tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between ab_2tdm_10 and ab_2tdm_10_debug is too large: {np.linalg.norm(ab_2tdm_10 - ab_2tdm_10_debug):.12f}."
    rdm2_sf_01 = ci_solver.make_sf_2rdm(0, 1)
    rdm2_sf_01_debug = ci_solver.ci_sigma_builder.sf_2rdm_debug(ci_0_det, ci_1_det)
    assert (
        np.linalg.norm(rdm2_sf_01 - rdm2_sf_01_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_sf_01 and rdm2_sf_01_debug is too large: {np.linalg.norm(rdm2_sf_01 - rdm2_sf_01_debug):.12f}."
    rdm2_sf_10 = ci_solver.make_sf_2rdm(1, 0)
    rdm2_sf_10_debug = ci_solver.ci_sigma_builder.sf_2rdm_debug(ci_1_det, ci_0_det)
    assert (
        np.linalg.norm(rdm2_sf_10 - rdm2_sf_10_debug) < rdm_threshold
    ), f"Norm of the difference between rdm2_sf_10 and rdm2_sf_10_debug is too large: {np.linalg.norm(rdm2_sf_10 - rdm2_sf_10_debug):.12f}."

    # Test the 3-RDMs

    aaa_3rdm_0, aab_3rdm_0, abb_3rdm_0, bbb_3rdm_0 = ci_solver.make_sd_3rdm(0, 0)
    aaa_3rdm_1, aab_3rdm_1, abb_3rdm_1, bbb_3rdm_1 = ci_solver.make_sd_3rdm(1, 1)

    aaa_3rdm_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(ci_0_det, ci_0_det, True)
    assert (
        np.linalg.norm(aaa_3rdm_0 - aaa_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aaa_3rdm_0 and aaa_3rdm_debug is too large: {np.linalg.norm(aaa_3rdm_0 - aaa_3rdm_debug):.12f}."

    bbb_3rdm_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_0_det, ci_0_det, False
    )
    assert (
        np.linalg.norm(bbb_3rdm_0 - bbb_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between bbb_3rdm_0 and bbb_3rdm_debug is too large: {np.linalg.norm(bbb_3rdm_0 - bbb_3rdm_debug):.12f}."

    aab_3rdm_debug = ci_solver.ci_sigma_builder.aab_3rdm_debug(ci_0_det, ci_0_det)
    assert (
        np.linalg.norm(aab_3rdm_0 - aab_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aab_3rdm_0 and aab_3rdm_debug is too large: {np.linalg.norm(aab_3rdm_0 - aab_3rdm_debug):.12f}."

    abb_3rdm_debug = ci_solver.ci_sigma_builder.abb_3rdm_debug(ci_0_det, ci_0_det)
    assert (
        np.linalg.norm(abb_3rdm_0 - abb_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between abb_3rdm_0 and abb_3rdm_debug is too large: {np.linalg.norm(abb_3rdm_0 - abb_3rdm_debug):.12f}."

    sf_3rdm = ci_solver.make_sf_3rdm(0)
    sf_3rdm_debug = ci_solver.ci_sigma_builder.sf_3rdm_debug(ci_0_det, ci_0_det)
    assert (
        np.linalg.norm(sf_3rdm - sf_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between sf_3rdm and sf_3rdm_debug is too large: {np.linalg.norm(sf_3rdm - sf_3rdm_debug):.12f}."

    aaa_3rdm_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(ci_1_det, ci_1_det, True)
    assert (
        np.linalg.norm(aaa_3rdm_1 - aaa_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aaa_3rdm_1 and aaa_3rdm_debug is too large: {np.linalg.norm(aaa_3rdm_1 - aaa_3rdm_debug):.12f}."

    bbb_3rdm_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_1_det, ci_1_det, False
    )
    assert (
        np.linalg.norm(bbb_3rdm_1 - bbb_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between bbb_3rdm_1 and bbb_3rdm_debug is too large: {np.linalg.norm(bbb_3rdm_1 - bbb_3rdm_debug):.12f}."

    aab_3rdm_debug = ci_solver.ci_sigma_builder.aab_3rdm_debug(ci_1_det, ci_1_det)
    assert (
        np.linalg.norm(aab_3rdm_1 - aab_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between aab_3rdm_1 and aab_3rdm_debug is too large: {np.linalg.norm(aab_3rdm_1 - aab_3rdm_debug):.12f}."

    abb_3rdm_debug = ci_solver.ci_sigma_builder.abb_3rdm_debug(ci_1_det, ci_1_det)
    assert (
        np.linalg.norm(abb_3rdm_1 - abb_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between abb_3rdm_1 and abb_3rdm_debug is too large: {np.linalg.norm(abb_3rdm_1 - abb_3rdm_debug):.12f}."

    sf_3rdm = ci_solver.make_sf_3rdm(1)
    sf_3rdm_debug = ci_solver.ci_sigma_builder.sf_3rdm_debug(ci_1_det, ci_1_det)
    assert (
        np.linalg.norm(sf_3rdm - sf_3rdm_debug) < rdm_threshold
    ), f"Norm of the difference between sf_3rdm and sf_3rdm_debug is too large: {np.linalg.norm(sf_3rdm - sf_3rdm_debug):.12f}."

    # Test the 3-TDMs
    aaa_3tdm_01, aab_3tdm_01, abb_3tdm_01, bbb_3tdm_01 = ci_solver.make_sd_3rdm(0, 1)
    aaa_3tdm_10, aab_3tdm_10, abb_3tdm_10, bbb_3tdm_10 = ci_solver.make_sd_3rdm(1, 0)
    aaa_3tdm_01_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_0_det, ci_1_det, True
    )
    assert (
        np.linalg.norm(aaa_3tdm_01 - aaa_3tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between aaa_3tdm_01 and aaa_3tdm_01_debug is too large: {np.linalg.norm(aaa_3tdm_01 - aaa_3tdm_01_debug):.12f}."
    bbb_3tdm_01_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_0_det, ci_1_det, False
    )
    assert (
        np.linalg.norm(bbb_3tdm_01 - bbb_3tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between bbb_3tdm_01 and bbb_3tdm_01_debug is too large: {np.linalg.norm(bbb_3tdm_01 - bbb_3tdm_01_debug):.12f}."
    aab_3tdm_01_debug = ci_solver.ci_sigma_builder.aab_3rdm_debug(ci_0_det, ci_1_det)
    assert (
        np.linalg.norm(aab_3tdm_01 - aab_3tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between aab_3tdm_01 and aab_3tdm_01_debug is too large: {np.linalg.norm(aab_3tdm_01 - aab_3tdm_01_debug):.12f}."
    abb_3tdm_01_debug = ci_solver.ci_sigma_builder.abb_3rdm_debug(ci_0_det, ci_1_det)
    assert (
        np.linalg.norm(abb_3tdm_01 - abb_3tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between abb_3tdm_01 and abb_3tdm_01_debug is too large: {np.linalg.norm(abb_3tdm_01 - abb_3tdm_01_debug):.12f}."
    aaa_3tdm_10_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_1_det, ci_0_det, True
    )
    assert (
        np.linalg.norm(aaa_3tdm_10 - aaa_3tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between aaa_3tdm_10 and aaa_3tdm_10_debug is too large: {np.linalg.norm(aaa_3tdm_10 - aaa_3tdm_10_debug):.12f}."
    bbb_3tdm_10_debug = ci_solver.ci_sigma_builder.aaa_3rdm_debug(
        ci_1_det, ci_0_det, False
    )
    assert (
        np.linalg.norm(bbb_3tdm_10 - bbb_3tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between bbb_3tdm_10 and bbb_3tdm_10_debug is too large: {np.linalg.norm(bbb_3tdm_10 - bbb_3tdm_10_debug):.12f}."
    aab_3tdm_10_debug = ci_solver.ci_sigma_builder.aab_3rdm_debug(ci_1_det, ci_0_det)
    assert (
        np.linalg.norm(aab_3tdm_10 - aab_3tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between aab_3tdm_10 and aab_3tdm_10_debug is too large: {np.linalg.norm(aab_3tdm_10 - aab_3tdm_10_debug):.12f}."
    abb_3tdm_10_debug = ci_solver.ci_sigma_builder.abb_3rdm_debug(ci_1_det, ci_0_det)
    assert (
        np.linalg.norm(abb_3tdm_10 - abb_3tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between abb_3tdm_10 and abb_3tdm_10_debug is too large: {np.linalg.norm(abb_3tdm_10 - abb_3tdm_10_debug):.12f}."
    sf_3tdm_01 = ci_solver.make_sf_3rdm(0, 1)
    sf_3tdm_01_debug = ci_solver.ci_sigma_builder.sf_3rdm_debug(ci_0_det, ci_1_det)
    assert (
        np.linalg.norm(sf_3tdm_01 - sf_3tdm_01_debug) < rdm_threshold
    ), f"Norm of the difference between sf_3tdm_01 and sf_3tdm_01_debug is too large: {np.linalg.norm(sf_3tdm_01 - sf_3tdm_01_debug):.12f}."
    sf_3tdm_10 = ci_solver.make_sf_3rdm(1, 0)
    sf_3tdm_10_debug = ci_solver.ci_sigma_builder.sf_3rdm_debug(ci_1_det, ci_0_det)
    assert (
        np.linalg.norm(sf_3tdm_10 - sf_3tdm_10_debug) < rdm_threshold
    ), f"Norm of the difference between sf_3tdm_10 and sf_3tdm_10_debug is too large: {np.linalg.norm(sf_3tdm_10 - sf_3tdm_10_debug):.12f}."

    # Test the spin-free cumulants

    sf_2cumulant = ci_solver.make_sf_2cumulant(0)
    sf_2cumulant_debug = ci_solver.ci_sigma_builder.sf_2cumulant_debug(
        ci_0_det, ci_0_det
    )
    assert (
        np.linalg.norm(sf_2cumulant - sf_2cumulant_debug) < rdm_threshold
    ), f"Norm of the difference between sf_2cumulant and sf_2cumulant_debug is too large: {np.linalg.norm(sf_2cumulant - sf_2cumulant_debug):.12f}."

    sf_3cumulant = ci_solver.make_sf_3cumulant(0)
    sf_3cumulant_debug = ci_solver.ci_sigma_builder.sf_3cumulant_debug(
        ci_0_det, ci_0_det
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
        nroots=2,
    )(rhf)
    ci.run()
    compare_rdms(ci)

    assert ci.E[0] == approx(-100.019788438077)


def test_ci_rdms_sa():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        states=[
            State(nel=10, multiplicity=1, ms=0.0),
            State(nel=10, multiplicity=3, ms=1.0),
        ],
        nroots=[2, 1],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        do_test_rdms=True,
    )(rhf)
    ci.run()
    compare_rdms(ci)

    assert ci.E[0] == approx(-100.01978843799819)
    assert ci.E[1] == approx(-99.68758394141096)
    assert ci.E[2] == approx(-99.7052645828813)

    g1 = ci.make_average_1rdm()
    l2 = ci.make_average_2cumulant()
    e_avg = ci.compute_average_energy()
    ci_ints = ci.sub_solvers[0].ints

    e_from_cumulants = ci_ints.E
    e_from_cumulants += np.einsum("pq,pq->", ci_ints.H, g1)
    e_from_cumulants += 0.5 * np.einsum("pqrs,pqrs->", ci_ints.V, l2)
    e_from_cumulants += 0.5 * np.einsum("pqrs,pr,qs->", ci_ints.V, g1, g1)
    e_from_cumulants -= 0.25 * np.einsum("pqrs,ps,qr->", ci_ints.V, g1, g1)

    assert e_avg == approx(e_from_cumulants)
