import numpy as np
import pytest

from forte2 import AVAS, System, RHF, UHF, CI, State
from forte2.mp import RMP2, UMP2
from forte2.props import (
    MutualCorrelationAnalysis,
    RMP2MPQOnTheFly,
    UMP2MPQOnTheFly,
    rmp2_mpq_onthefly_no,
    ump2_mpq_onthefly_no,
)
from forte2.helpers.comparisons import approx
from forte2.base_classes import DavidsonLiuParams


def test_mutual_correlation_h2_singlet():
    """Test mutual correlation analysis on H2 molecule in STO-6G basis at dissociation."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 10.0
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=[0, 1],
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
    )(rhf)
    ci.run()

    mca = MutualCorrelationAnalysis(ci, root=0, sub_solver_index=0)

    # verify some known values for H2 in STO-6G at dissociation
    assert mca.total_correlation == approx(0.875)
    assert mca.M2[0, 1] == approx(0.75)
    assert mca.M2[1, 0] == approx(0.75)
    assert mca.M2[0, 0] == approx(0.0)
    assert mca.M2[1, 1] == approx(0.0)


def test_mutual_correlation_h2_triplet_lowspin():
    """Test mutual correlation analysis on H2 molecule in the triplet low-spin (ms=0) state in STO-6G basis at dissociation."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 10.0
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=3, ms=0.0),
        active_orbitals=[0, 1],
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
    )(rhf)
    ci.run()

    mca = MutualCorrelationAnalysis(ci, root=0, sub_solver_index=0)

    # verify some known values for H2 in STO-6G at dissociation
    assert mca.total_correlation == approx(0.875)
    assert mca.M2[0, 1] == approx(0.75)
    assert mca.M2[1, 0] == approx(0.75)
    assert mca.M2[0, 0] == approx(0.0)
    assert mca.M2[1, 1] == approx(0.0)


def test_mutual_correlation_h2_triplet_highspin():
    """Test mutual correlation analysis on H2 molecule in the triplet high-spin state (multiplicity=3, ms=1.0) in STO-6G basis at dissociation."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 10.0
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=3, ms=1.0),
        active_orbitals=[0, 1],
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
    )(rhf)
    ci.run()

    mca = MutualCorrelationAnalysis(ci, root=0, sub_solver_index=0)

    # verify some known values for H2 in STO-6G at dissociation
    assert mca.total_correlation == approx(0.0)
    assert mca.M2[0, 1] == approx(0.0)
    assert mca.M2[1, 0] == approx(0.0)
    assert mca.M2[0, 0] == approx(0.0)
    assert mca.M2[1, 1] == approx(0.0)


def test_mutual_correlation_h2_orbopt():
    """Test mutual correlation analysis on H2 molecule in cc-pVDZ basis at 2.0 Angstroms separation."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=list(range(10)),
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
    )(rhf)
    ci.run()

    mca = MutualCorrelationAnalysis(ci)
    assert mca.total_correlation == approx(0.512615148)
    assert mca.M2[0, 1] == approx(0.416025017)

    # Use a fixed seed for deterministic optimization in tests
    mca.optimize_orbitals(seed=1023)
    assert mca.total_correlation == approx(0.512615148)
    assert mca.M2[0, 1] == approx(0.511668631)


def test_mutual_correlation_h6():
    """Test mutual correlation analysis on H6 and the sto-3g basis."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    H 0.0 0.0 6.0
    """

    system = System(xyz=xyz, basis_set="sto-3g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0),
        active_orbitals=list(range(6)),
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
    )(rhf)
    ci.run()

    mca = MutualCorrelationAnalysis(ci)
    assert mca.total_correlation == approx(0.815410515)
    assert mca.M2[2, 3] == approx(0.562132887)

    summary = mca.mutual_correlation_matrix_summary()
    assert float(summary.splitlines()[5].split()[-1]) == approx(0.562133)


def test_rmp2_mpq_first_order_and_avas():
    """Test first-order RMP2-MPQ elements and AVAS orbital selection."""

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")
    rhf = RHF(charge=0)(system)
    mp2 = RMP2(store_t2=True)(rhf)
    mp2.run()

    gamma1 = mp2.make_1rdm()
    first_order_mpq = RMP2MPQOnTheFly(
        mp2, U=np.eye(mp2.nocc + mp2.nvir), include_quadratic=False
    )
    i, j, a, b = 0, 1, 0, 1
    assert first_order_mpq.lambda2_ab_first_order_elem(
        i, j, mp2.nocc + a, mp2.nocc + b
    ) == approx(mp2.t2[i, j, a, b])
    assert first_order_mpq.lambda2_aa_first_order_elem(
        i, j, mp2.nocc + a, mp2.nocc + b
    ) == approx(mp2.t2[i, j, a, b] - mp2.t2[i, j, b, a])

    avas = AVAS(
        subspace=["O(2p)"], selection_method="total", num_active=3
    )(rhf)
    avas_mpq = rmp2_mpq_onthefly_no(mp2, avas=avas)
    assert avas.executed
    assert not avas_mpq.include_quadratic
    assert avas_mpq.rdm_info_selection == "avas"
    assert len(avas_mpq.rdm_info_indices) == avas.nactv
    weights = avas_mpq.rdm_info_selection_details[
        "avas_projection_weights"
    ]
    expected = tuple(
        sorted(np.argsort(-weights, kind="stable")[: avas.nactv].tolist())
    )
    assert avas_mpq.rdm_info_indices == expected
    expected_gamma1_no = avas_mpq.U.T @ gamma1 @ avas_mpq.U
    assert avas_mpq.Gamma1_mo == approx(gamma1)
    assert avas_mpq.Gamma1_no == approx(expected_gamma1_no)
    assert avas_mpq.Gamma1 == approx(expected_gamma1_no)
    assert avas_mpq.Γ1 == approx(expected_gamma1_no)
    assert avas_mpq.occs == approx(np.diag(expected_gamma1_no))
    assert avas_mpq.C_no == approx(mp2.C_no)
    assert avas_mpq.no_occs == approx(mp2.no_occs)
    assert avas_mpq.U == approx(mp2.U_no)


def test_ump2_mpq_first_order_and_optional_quadratic_terms():
    """Test first-order UMP2-MPQ elements and optional quadratic terms."""

    euhf = -76.0217659883263
    emp2 = -76.221819034
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    uhf = UHF(charge=0, ms=0)(system)
    mp2 = UMP2(store_t2=True)(uhf)
    mp2.run()

    identity = np.eye(mp2.nmo)
    first_order = UMP2MPQOnTheFly(
        mp2, Ua=identity, Ub=identity, include_quadratic=False
    )
    full = UMP2MPQOnTheFly(
        mp2, Ua=identity, Ub=identity, include_quadratic=True
    )

    a = mp2.naocc
    b = mp2.nbocc
    assert first_order.lambda2_aa_first_order_elem(0, 1, a, a + 1) == approx(
        mp2.t2_a[0, 1, 0, 1]
    )
    assert first_order.lambda2_bb_first_order_elem(0, 1, b, b + 1) == approx(
        mp2.t2_b[0, 1, 0, 1]
    )
    assert first_order.lambda2_ab_elem(0, 0, a, b) == approx(
        mp2.t2_ab[0, 0, 0, 0]
    )
    assert full.lambda2_ab_elem(0, b, 0, b) == approx(
        full.lambda2_ab_first_order_elem(0, b, 0, b)
        + full.lambda2_ab_quadratic_elem(0, b, 0, b)
    )
    assert abs(full.lambda2_ab_quadratic_elem(0, b, 0, b)) > 0.0

    assert uhf.E == approx(euhf)
    assert mp2.E_total == approx(emp2)


def test_ump2_mpq_wrapper():
    """Test UMP2-MPQ natural orbitals and RDM-info selection."""

    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.4
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    uhf = UHF(charge=0, ms=0)(system)
    mp2 = UMP2(store_t2=True)(uhf)
    mp2.run()

    gamma1 = mp2.make_1rdm_sd()
    no_transform = mp2.make_natural_orbital_transform(gamma1)
    C_no, occupations, Ua, Ub = no_transform
    gamma1_no_a, gamma1_no_b = mp2.make_1rdm_no_sd(gamma1, no_transform)
    gamma1_no = mp2.make_1rdm_no_sf(gamma1, no_transform)

    mpq = ump2_mpq_onthefly_no(mp2, mo_range=(0, 2))
    assert isinstance(mpq, UMP2MPQOnTheFly)
    assert not mpq.include_quadratic
    assert mpq.C_no == approx(C_no)
    assert mpq.no_occs == approx(occupations)
    assert mpq.Ua == approx(Ua)
    assert mpq.Ub == approx(Ub)
    assert mpq.gamma1_mo_a == approx(gamma1[0])
    assert mpq.gamma1_mo_b == approx(gamma1[1])
    assert mpq.gamma1_a == approx(gamma1_no_a)
    assert mpq.gamma1_b == approx(gamma1_no_b)
    assert mpq.γa == approx(gamma1_no_a)
    assert mpq.γb == approx(gamma1_no_b)
    assert mpq.Gamma1_no == approx(gamma1_no)
    assert mpq.Gamma1 == approx(gamma1_no)
    assert mpq.Γ1 == approx(gamma1_no)
    assert mpq.occs == approx(occupations)
    assert mpq.rdm_info_indices == (0, 1)
    assert mpq.rdm_info_selection == "mo_range"

    M1 = mpq.make_M1()
    M2 = mpq.make_M2()
    assert M1.shape == (mp2.nmo,)
    assert M2.shape == (mp2.nmo, mp2.nmo)
    assert np.all(M1 >= -1e-12)
    assert np.count_nonzero(M1[2:]) == 0
    assert np.count_nonzero(M2[2:, :]) == 0
    assert np.count_nonzero(M2[:, 2:]) == 0

    occupation_mpq = ump2_mpq_onthefly_no(
        mp2, occupation_window=(0.02, 1.98)
    )
    expected = tuple(
        np.flatnonzero((occupations >= 0.02) & (occupations <= 1.98)).tolist()
    )
    assert occupation_mpq.rdm_info_indices == expected
    assert occupation_mpq.rdm_info_selection == "natural_occupation"

    avas = AVAS(subspace=["H(1s)"])(RHF(charge=0)(system))
    with pytest.raises(TypeError, match="requires restricted orbitals"):
        ump2_mpq_onthefly_no(mp2, avas=avas)

    with pytest.raises(ValueError, match="Choose only one"):
        ump2_mpq_onthefly_no(
            mp2, mo_range=(0, 2), occupation_window=(0.02, 1.98)
        )
