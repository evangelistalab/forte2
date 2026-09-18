import numpy as np
import pytest

from forte2 import AVAS, System, RHF, UHF, CI, State
from forte2.mp import RMP2, UMP2
from forte2.props import (
    MutualCorrelationAnalysis,
    RMP2MPQOnTheFly,
    UMP2MPQOnTheFly,
    suggest_mutual_correlation_active_spaces,
    rmp2_mpq_onthefly_no,
    ump2_mpq_onthefly_no,
)
from forte2.helpers.comparisons import approx
from forte2.base_classes import DavidsonLiuParams


def test_significant_score_selection_completes_degenerate_groups():
    """Threshold significant scores without splitting degenerate NOs."""
    matrix = np.zeros((5, 5))
    matrix[0, 1] = matrix[1, 0] = 0.10
    matrix[1, 2] = matrix[2, 1] = 0.01
    matrix[3, 4] = matrix[4, 3] = 5.0e-4
    occupations = np.array([1.8, 0.2, 1.0, 1.8 + 5.0e-11, 0.0])

    result = suggest_mutual_correlation_active_spaces(
        matrix,
        occupations,
        absolute_threshold=1.0e-3,
        mandatory_indices=[4],
    )

    assert result["significant_edges"] == (
        (0, 1, 0.10),
        (1, 2, 0.01),
    )
    assert result["significant_scores"] == approx(
        np.array([0.10, 0.11, 0.01, 0.0, 0.0])
    )
    assert result["suggestions"][0.15]["score_selected_indices"] == (0, 1)
    assert result["suggestions"][0.15]["degeneracy_completed_indices"] == (3,)
    assert result["suggestions"][0.15]["active_indices"] == (0, 1, 3, 4)
    assert result["suggestions"][0.05]["active_indices"] == (0, 1, 2, 3, 4)
    assert result["suggestions"][0.0]["active_indices"] == (0, 1, 2, 3, 4)


def test_significant_score_selection_with_no_edges_keeps_only_mandatory_group():
    """A zero matrix must not make an eta=0 threshold select all orbitals."""
    result = suggest_mutual_correlation_active_spaces(
        np.zeros((3, 3)),
        np.array([1.0, 1.0 + 5.0e-11, 0.0]),
        mandatory_indices=[0],
    )

    assert result["maximum_significant_score"] == 0.0
    for suggestion in result["suggestions"].values():
        assert suggestion["score_selected_indices"] == ()
        assert suggestion["degeneracy_completed_indices"] == (1,)
        assert suggestion["active_indices"] == (0, 1)


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

    selected = (i, j, mp2.nocc + a, mp2.nocc + b)
    quadratic_mpq = RMP2MPQOnTheFly(
        mp2,
        U=np.eye(mp2.nocc + mp2.nvir),
        include_quadratic=True,
        orbital_indices=selected,
    )
    t2_as = mp2.t2 - mp2.t2.transpose(0, 1, 3, 2)
    expected_oooo = 0.5 * np.einsum(
        "ab,ab->", t2_as[i, j].conj(), t2_as[i, j], optimize=True
    )
    expected_vvvv = 0.5 * np.einsum(
        "ij,ij->",
        t2_as[:, :, a, b].conj(),
        t2_as[:, :, a, b],
        optimize=True,
    )
    expected_ovov = -np.einsum(
        "mc,mc->",
        mp2.t2[i, :, :, b].conj(),
        mp2.t2[i, :, :, a],
        optimize=True,
    )
    expected_vovo = -np.einsum(
        "mc,mc->",
        mp2.t2[:, i, b, :].conj(),
        mp2.t2[:, i, a, :],
        optimize=True,
    )
    assert quadratic_mpq.lambda2_aa_quadratic_elem(i, j, i, j) == approx(
        expected_oooo
    )
    assert quadratic_mpq.lambda2_aa_quadratic_elem(
        mp2.nocc + a,
        mp2.nocc + b,
        mp2.nocc + a,
        mp2.nocc + b,
    ) == approx(expected_vvvv)
    assert quadratic_mpq.lambda2_ab_quadratic_elem(i, mp2.nocc + a, i, mp2.nocc + b) == approx(
        expected_ovov
    )
    assert quadratic_mpq.lambda2_ab_quadratic_elem(mp2.nocc + a, i, mp2.nocc + b, i) == approx(
        expected_vovo
    )

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
    assert avas_mpq.C_no == approx(mp2.C[0] @ avas_mpq.U)
    assert avas_mpq.no_occs == approx(np.diag(expected_gamma1_no))


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


def test_ump2_exact_selected_common_no_transform():
    """Compare exact selected-space blocks against dense rank-four rotations."""

    xyz = """
    O  0.000000000000  0.000000000000 -0.061664597388
    H  0.000000000000 -0.711620616369  0.489330954643
    H  0.000000000000  0.711620616369  0.489330954643
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    uhf = UHF(charge=0, ms=1)(system)
    mp2 = UMP2(store_t2=True)(uhf)
    mp2.run()
    assert mp2.naocc != mp2.nbocc

    rng = np.random.default_rng(7)
    Ua = np.linalg.qr(
        rng.normal(size=(mp2.nmo, mp2.nmo))
        + 1j * rng.normal(size=(mp2.nmo, mp2.nmo))
    )[0]
    Ub = np.linalg.qr(
        rng.normal(size=(mp2.nmo, mp2.nmo))
        + 1j * rng.normal(size=(mp2.nmo, mp2.nmo))
    )[0]
    selected = (0, 2, 4)
    analyzer = UMP2MPQOnTheFly(
        mp2,
        Ua=Ua,
        Ub=Ub,
        orbital_indices=selected,
        common_no_transform="exact_selected",
        include_quadratic=True,
    )
    analyzer.make_measures()

    def canonical_first_order(t2, nocc_first, nocc_second=None):
        if nocc_second is None:
            nocc_second = nocc_first
        tensor = np.zeros((mp2.nmo,) * 4)
        tensor[
            :nocc_first, :nocc_second, nocc_first:, nocc_second:
        ] = t2
        tensor[
            nocc_first:, nocc_second:, :nocc_first, :nocc_second
        ] = t2.transpose(2, 3, 0, 1)
        return tensor

    lambda_aa_mo = canonical_first_order(mp2.t2_a, mp2.naocc)
    lambda_bb_mo = canonical_first_order(mp2.t2_b, mp2.nbocc)
    lambda_ab_mo = canonical_first_order(
        mp2.t2_ab, mp2.naocc, mp2.nbocc
    )

    lambda_aa_quadratic_mo = np.zeros((mp2.nmo,) * 4)
    lambda_aa_quadratic_mo[
        : mp2.naocc, : mp2.naocc, : mp2.naocc, : mp2.naocc
    ] = 0.5 * np.einsum(
        "ijab,klab->ijkl", mp2.t2_a.conj(), mp2.t2_a, optimize=True
    )
    lambda_aa_quadratic_mo[
        mp2.naocc :, mp2.naocc :, mp2.naocc :, mp2.naocc :
    ] = 0.5 * np.einsum(
        "ijab,ijcd->abcd", mp2.t2_a.conj(), mp2.t2_a, optimize=True
    )

    lambda_bb_quadratic_mo = np.zeros((mp2.nmo,) * 4)
    lambda_bb_quadratic_mo[
        : mp2.nbocc, : mp2.nbocc, : mp2.nbocc, : mp2.nbocc
    ] = 0.5 * np.einsum(
        "ijab,klab->ijkl", mp2.t2_b.conj(), mp2.t2_b, optimize=True
    )
    lambda_bb_quadratic_mo[
        mp2.nbocc :, mp2.nbocc :, mp2.nbocc :, mp2.nbocc :
    ] = 0.5 * np.einsum(
        "ijab,ijcd->abcd", mp2.t2_b.conj(), mp2.t2_b, optimize=True
    )

    lambda_ab_quadratic_mo = np.zeros((mp2.nmo,) * 4)
    lambda_ab_quadratic_mo[
        : mp2.naocc, mp2.nbocc :, : mp2.naocc, mp2.nbocc :
    ] = -np.einsum(
        "imcb,jmca->iajb", mp2.t2_ab.conj(), mp2.t2_ab, optimize=True
    )
    lambda_ab_quadratic_mo[
        mp2.naocc :, : mp2.nbocc, mp2.naocc :, : mp2.nbocc
    ] = -np.einsum(
        "mibc,mjac->aibj", mp2.t2_ab.conj(), mp2.t2_ab, optimize=True
    )

    Ua_selected = Ua[:, selected]
    Ub_selected = Ub[:, selected]

    def transform(tensor, U1, U2):
        return np.einsum(
            "pP,qQ,pqrs,rR,sS->PQRS",
            U1.conj(),
            U2.conj(),
            tensor,
            U1,
            U2,
            optimize=True,
        )

    expected_aa = transform(lambda_aa_mo, Ua_selected, Ua_selected)
    expected_bb = transform(lambda_bb_mo, Ub_selected, Ub_selected)
    expected_ab = transform(lambda_ab_mo, Ua_selected, Ub_selected)
    expected_aa_quadratic = transform(
        lambda_aa_quadratic_mo, Ua_selected, Ua_selected
    )
    expected_bb_quadratic = transform(
        lambda_bb_quadratic_mo, Ub_selected, Ub_selected
    )
    expected_ab_quadratic = transform(
        lambda_ab_quadratic_mo, Ua_selected, Ub_selected
    )

    assert analyzer.lambda2_aa_selected == approx(expected_aa)
    assert analyzer.lambda2_bb_selected == approx(expected_bb)
    assert analyzer.lambda2_ab_selected == approx(expected_ab)
    assert analyzer.lambda2_aa_quadratic_selected == approx(
        expected_aa_quadratic
    )
    assert analyzer.lambda2_bb_quadratic_selected == approx(
        expected_bb_quadratic
    )
    assert analyzer.lambda2_ab_quadratic_selected == approx(
        expected_ab_quadratic
    )
    assert analyzer.lambda2_selected_indices == selected
    assert analyzer.lambda2_aa_selected.shape == (3, 3, 3, 3)
    assert analyzer.exact_selected_tensor_memory_mb < 0.01
    unselected = sorted(set(range(mp2.nmo)) - set(selected))
    assert np.count_nonzero(analyzer.M1[unselected]) == 0
    assert np.count_nonzero(analyzer.M2[unselected, :]) == 0
    assert np.count_nonzero(analyzer.M2[:, unselected]) == 0

    with pytest.raises(MemoryError, match="estimated"):
        UMP2MPQOnTheFly(
            mp2,
            Ua=Ua,
            Ub=Ub,
            orbital_indices=selected,
            common_no_transform="exact_selected",
            max_exact_tensor_memory_mb=1.0e-6,
        )

    with pytest.raises(ValueError, match="exceeds max_exact_orbitals=2"):
        UMP2MPQOnTheFly(
            mp2,
            Ua=Ua,
            Ub=Ub,
            orbital_indices=selected,
            common_no_transform="exact_selected",
            max_exact_orbitals=2,
        )


def test_ump2_mpq_wrapper():
    """Test UMP2-MPQ natural orbitals and RDM-info selection."""

    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.4
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    uhf = UHF(charge=0, ms=0)(system)
    mp2 = UMP2(store_t2=False)(uhf)
    mp2.run()

    gamma1 = mp2.make_1rdm_sd()
    mpq = ump2_mpq_onthefly_no(mp2, mo_range=(0, 2))
    C_no, occupations, Ua, Ub = mpq.no_transform
    gamma1_no_a = Ua.T @ gamma1[0] @ Ua
    gamma1_no_b = Ub.T @ gamma1[1] @ Ub
    gamma1_no = gamma1_no_a + gamma1_no_b

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
    assert C_no.T @ system.ints_overlap() @ C_no == approx(np.eye(mp2.nmo))
    assert mpq.rdm_info_indices == (0, 1)
    assert mpq.rdm_info_selection == "mo_range"
    assert mp2.t2_a is None
    assert mp2.t2_b is None
    assert mp2.t2_ab is None

    M1 = mpq.make_M1()
    M2 = mpq.make_M2()
    assert M1.shape == (mp2.nmo,)
    assert M2.shape == (mp2.nmo, mp2.nmo)
    assert np.all(M1 >= -1e-12)
    assert np.count_nonzero(M1[2:]) == 0
    assert np.count_nonzero(M2[2:, :]) == 0
    assert np.count_nonzero(M2[:, 2:]) == 0

    exact_mpq = ump2_mpq_onthefly_no(
        mp2,
        mo_range=(0, 2),
        common_no_transform="exact_selected",
    )
    exact_M1, exact_M2 = exact_mpq.make_measures()
    assert exact_mpq.common_no_transform == "exact_selected"
    assert exact_mpq.lambda2_aa_selected.shape == (2, 2, 2, 2)
    assert exact_M1.shape == (mp2.nmo,)
    assert exact_M2.shape == (mp2.nmo, mp2.nmo)
    assert mp2.t2_a is None
    assert mp2.t2_b is None
    assert mp2.t2_ab is None

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
