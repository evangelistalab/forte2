import numpy as np
import pytest

from forte2 import System, State, MOSpace
from forte2.lib import rdms
from forte2.lib.sparse_ops import SparseState
from forte2.lib.det import Determinant
from forte2.lib.ci_helpers import CIStrings, SelectedCIHelper
from forte2.ci import CI
from forte2.scf import RHF
from forte2.sci import SelectedCI
from forte2.sci.sci import _SelectedCISingleStateSolver
from forte2.helpers.comparisons import approx
from forte2.base_classes.params import SelectedCIParams, DavidsonLiuParams


def _h4_rhf():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    """
    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    return RHF(charge=0, e_tol=1e-14)(system)


def test_sci1():
    """Test that SelectedCI reproduces the FCI energy on 4 H atoms in a chain with STO-6G basis set."""

    efci = -2.180967812920

    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            num_batches_per_thread=16,
        ),
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(efci)


def test_sci_pinned_only_guess():
    efci = -2.180967812920

    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            guess_dets=[],
            pinned_guess_dets=[Determinant("22")],
            num_batches_per_thread=16,
        ),
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(efci)


def test_sci2():
    """Test SelectedCI with a single determinant guess."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-4,
            pt2_threshold=0.0,
            guess_occ_window=0,
            guess_vir_window=0,
            pt2_regularizer="dsrg",
            pt2_regularizer_strength=0.2,
            num_batches_per_thread=16,
        ),
        nroots=1,
        do_test_rdms=True,
    )(rhf)

    sci.run()

    # this is the variational energy
    assert sci.E[0] == pytest.approx(-3.321294103198, abs=1e-9)
    # this is the regularized PT2 correction
    assert sci.E_pt2[0] == pytest.approx(-2.53555293e-05, abs=1e-9)


def test_sci3():
    """Test SelectedCI on multiple states without spin penalty."""
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            guess_occ_window=2,
            guess_vir_window=2,
            do_spin_penalty=False,
            num_batches_per_thread=16,
        ),
        nroots=4,
        do_test_rdms=True,
    )(rhf)

    sci.run()

    assert sci.E[0] == pytest.approx(-3.3213220620, abs=1e-8)
    assert sci.E[3] == pytest.approx(-3.0403077216, abs=1e-8)


def test_sci4():
    """Test SelectedCI on multiple states with spin penalty."""
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        nroots=2,
        do_test_rdms=True,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            guess_occ_window=2,
            guess_vir_window=2,
            do_spin_penalty=True,
            num_batches_per_thread=16,
        ),
    )(rhf)

    sci.run()

    assert sci.E[0] == pytest.approx(-3.3213219202, abs=1e-8)
    assert sci.E[1] == pytest.approx(-3.0403076453, abs=1e-8)


@pytest.mark.slow
def test_sci5():
    """Test SelectedCI on a core-ionized state."""
    xyz = f"""
    Ne 0.0 0.0 0.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", cholesky_tei=True, cholesky_tol=1e-16)

    rhf = RHF(charge=0, e_tol=1e-10)(system)
    sci = SelectedCI(
        states=State(nel=9, multiplicity=2, ms=0.5),
        active_orbitals=list(range(14)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            guess_dets=[Determinant("a2222")],
            num_batches_per_thread=16,
        ),
        nroots=1,
    )(rhf)

    sci.run()
    assert sci.E[0] == pytest.approx(-96.60170826050286, abs=5e-7)


@pytest.mark.skip(reason="Could not reproduce with FCI with energy_shift")
def test_sci6():
    """Test SelectedCI on a core-excited state."""
    xyz = """
    Ne 0.0 0.0 0.0
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", cholesky_tei=True, cholesky_tol=1e-16)

    rhf = RHF(charge=0, e_tol=1e-10)(system)

    sci = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci_ref",
            var_threshold=3e-4,
            pt2_threshold=0.0,
            guess_dets=[Determinant("a2222b"), Determinant("b2222a")],
            do_spin_penalty=True,
            screening_criterion="hbci",
            num_batches_per_thread=16,
        ),
        nroots=1,
        die_if_not_converged=False,
    )(rhf)

    sci.run()

    assert sci.E[0] == approx(-95.67969625695353)


def test_sci_exact_algorithm():
    """FCI energy from exact selected-CI diagonalization."""
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            ci_algorithm="exact",
        ),
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(-2.180967812920)


def test_sci_make_sf_1rdm():
    """Spin-free 1-RDM should be available from the SCI helper-backed path."""
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )(rhf)
    sci.run()

    rdm1 = sci.sub_solvers[0].make_rdm(0, order=1, kind="sf")
    assert rdm1.shape == (4, 4)
    assert np.trace(rdm1) == pytest.approx(4.0, abs=1e-8)
    assert sci.E[0] == approx(-2.180967812920)


def test_sci_1trdm():
    """SelectedCI transition 1-RDMs match 1-RDMs for same state."""
    norb = 2
    dets = [Determinant("a0"), Determinant("0a")]
    c = np.array([[1.0, 0.0], [0.0, 1.0]])
    h = np.zeros((norb, norb))
    v = np.zeros((norb, norb, norb, norb))

    helper = SelectedCIHelper(norb, dets, c, 0.0, h, v, 0)

    assert np.allclose(helper.a_1trdm(helper, 0, 1), helper.a_1rdm(0, 1))
    assert np.allclose(helper.b_1trdm(helper, 0, 1), helper.b_1rdm(0, 1))
    assert np.allclose(helper.sf_1trdm(helper, 0, 1), helper.sf_1rdm(0, 1))


def test_sci_1trdm_allows_different_root_counts():
    """SelectedCI transition 1-RDMs between states with different root numbers."""
    norb = 2
    dets = [Determinant("20"), Determinant("ba")]
    h = np.zeros((norb, norb))
    v = np.zeros((norb, norb, norb, norb))

    left_c = np.array([[1.0], [0.0]])
    right_c = np.array([[1.0, 0.0], [0.0, 1.0]])
    left_helper = SelectedCIHelper(norb, dets, left_c, 0.0, h, v, 0)
    right_helper = SelectedCIHelper(norb, dets, right_c, 0.0, h, v, 0)

    assert np.allclose(
        left_helper.a_1trdm(right_helper, 0, 1), [[0.0, 1.0], [0.0, 0.0]]
    )
    assert np.allclose(
        right_helper.a_1trdm(left_helper, 1, 0), [[0.0, 0.0], [1.0, 0.0]]
    )
    assert np.allclose(left_helper.b_1trdm(right_helper, 0, 1), np.zeros((norb, norb)))
    assert np.allclose(
        left_helper.sf_1trdm(right_helper, 0, 1),
        left_helper.a_1trdm(right_helper, 0, 1)
        + left_helper.b_1trdm(right_helper, 0, 1),
    )


def test_sci_1trdm_matches_second_strings_between_different_spaces():
    """SelectedCI transition 1-RDMs match spectator strings across different spaces."""
    norb = 4
    h = np.zeros((norb, norb))
    v = np.zeros((norb, norb, norb, norb))

    left_dets = [
        Determinant("ab00"),
        Determinant("ba00"),
        Determinant("0ab0"),
        Determinant("0ba0"),
        Determinant("a00b"),
        Determinant("00ab"),
    ]
    right_dets = [
        Determinant("2000"),
        Determinant("ab00"),
        Determinant("0a0b"),
        Determinant("b0a0"),
        Determinant("00ab"),
        Determinant("a00b"),
    ]
    left_c = np.array([[0.3], [-0.5], [0.7], [1.1], [-0.2], [0.4]])
    right_c = np.array([[0.6], [-0.8], [0.5], [1.2], [-0.3], [0.9]])

    left_state = SparseState(dict(zip(left_dets, left_c[:, 0])))
    right_state = SparseState(dict(zip(right_dets, right_c[:, 0])))
    expected_a = rdms.compute_a_1rdm(left_state, right_state, norb)
    expected_b = rdms.compute_b_1rdm(left_state, right_state, norb)
    assert np.count_nonzero(np.abs(expected_a) > 1e-12) > 4
    assert np.count_nonzero(np.abs(expected_b) > 1e-12) > 4

    left_helper = SelectedCIHelper(norb, left_dets, left_c, 0.0, h, v, 0)
    right_helper = SelectedCIHelper(norb, right_dets, right_c, 0.0, h, v, 0)
    assert np.allclose(left_helper.a_1trdm(right_helper, 0, 0), expected_a)
    assert np.allclose(left_helper.b_1trdm(right_helper, 0, 0), expected_b)
    assert np.allclose(
        left_helper.sf_1trdm(right_helper, 0, 0), expected_a + expected_b
    )


def test_sci_1trdm_validates_helper_compatibility():
    """SelectedCI transition 1-RDMs validate helper compatibility."""
    h2 = np.zeros((2, 2))
    v2 = np.zeros((2, 2, 2, 2))
    h3 = np.zeros((3, 3))
    v3 = np.zeros((3, 3, 3, 3))

    helper = SelectedCIHelper(2, [Determinant("20")], np.array([[1.0]]), 0.0, h2, v2, 0)
    different_norb = SelectedCIHelper(
        3, [Determinant("200")], np.array([[1.0]]), 0.0, h3, v3, 0
    )
    different_electrons = SelectedCIHelper(
        2, [Determinant("a0")], np.array([[1.0]]), 0.0, h2, v2, 0
    )
    two_root_helper = SelectedCIHelper(
        2, [Determinant("20")], np.array([[1.0, 0.0]]), 0.0, h2, v2, 0
    )

    with pytest.raises(RuntimeError, match="number of MOs"):
        helper.a_1trdm(different_norb, 0, 0)

    with pytest.raises(RuntimeError, match="alpha and beta electrons"):
        helper.a_1trdm(different_electrons, 0, 0)

    with pytest.raises(RuntimeError, match="Root index out of range"):
        helper.a_1trdm(two_root_helper, 1, 0)

    with pytest.raises(RuntimeError, match="Root index out of range"):
        helper.a_1trdm(two_root_helper, 0, 2)


@pytest.mark.parametrize(
    ("window_occ", "window_vir", "message"),
    [
        (-1, 1, "guess_occ_window must be non-negative"),
        (1, -1, "guess_vir_window must be non-negative"),
        (3, 0, "guess_occ_window=3 is larger than the number of active"),
    ],
)
def test_sci_initial_guess_validates_occupation_windows(
    window_occ, window_vir, message
):
    """SelectedCI initial determinant guesses validate occupation windows."""
    solver = _SelectedCISingleStateSolver(
        sci_params=SelectedCIParams(),
        mo_space=MOSpace(nmo=4, active_orbitals=list(range(4))),
        state=State(nel=4, multiplicity=1, ms=0.0),
    )

    with pytest.raises(ValueError, match=message):
        solver._generate_initial_guess_dets(window_occ, window_vir)


def test_sci_transition_dipole_matches_ci():
    """SelectedCI transition dipoles should match CI in the full selected space."""
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    H 0.0 0.0 4.0
    H 0.0 0.0 5.0
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0)(system)
    ci = CI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(6)),
        nroots=4,
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        do_transition_dipole=True,
    )(rhf)
    ci.run()

    sci = SelectedCI(
        states=State(nel=6, multiplicity=1, ms=0.0),
        active_orbitals=list(range(6)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            screening_criterion="hbci",
            guess_occ_window=2,
            guess_vir_window=2,
        ),
        nroots=4,
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        do_transition_dipole=True,
    )(rhf)
    sci.run()

    assert np.allclose(sci.E, ci.E, atol=1e-8)
    assert sci.transition_dipoles.keys() == ci.transition_dipoles.keys()
    assert sci.oscillator_strengths.keys() == ci.oscillator_strengths.keys()

    for key in ci.transition_dipoles:
        assert np.abs(sci.transition_dipoles[key]) == pytest.approx(
            np.abs(ci.transition_dipoles[key]), abs=1e-4
        )
        assert sci.oscillator_strengths[key] == pytest.approx(
            ci.oscillator_strengths[key], abs=1e-4
        )


def test_sci_transition_dipole_different_nroots_matches_ci():
    """SelectedCI transition dipoles support state blocks with different nroots."""
    rhf = _h4_rhf()
    states = [
        State(nel=4, multiplicity=1, ms=0.0),
        State(nel=4, multiplicity=3, ms=0.0),
    ]
    davidson_liu_params = DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5)

    ci = CI(
        states=states,
        active_orbitals=list(range(4)),
        nroots=[1, 2],
        davidson_liu_params=davidson_liu_params,
        do_transition_dipole=True,
    )(rhf)
    ci.run()

    sci = SelectedCI(
        states=states,
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            screening_criterion="hbci",
        ),
        nroots=[1, 2],
        davidson_liu_params=davidson_liu_params,
        do_transition_dipole=True,
    )(rhf)
    sci.run()

    assert [solver.nroot for solver in sci.sub_solvers] == [1, 2]
    assert np.allclose(sci.E, ci.E, atol=1e-8)
    assert sci.transition_dipoles.keys() == ci.transition_dipoles.keys()
    assert sci.oscillator_strengths.keys() == ci.oscillator_strengths.keys()

    for key in [(0, 1), (0, 2)]:
        assert np.allclose(
            sci.make_rdm(*key, order=1, kind="sf"),
            ci.make_rdm(*key, order=1, kind="sf"),
            atol=1e-6,
        )

    for key in ci.transition_dipoles:
        assert np.abs(sci.transition_dipoles[key]) == pytest.approx(
            np.abs(ci.transition_dipoles[key]), abs=1e-4
        )
        assert sci.oscillator_strengths[key] == pytest.approx(
            ci.oscillator_strengths[key], abs=1e-4
        )


def test_sci_make_rdms():
    """Test 1-/2-RDMs and the 2-cumulant from SelectedCI."""
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            guess_dets=[Determinant("22")],
        ),
    )(rhf)
    sci.run()

    ci = CI(
        states=State(nel=4, multiplicity=1, ms=0.0), active_orbitals=list(range(4))
    )(rhf)
    ci.run()

    # Test the 1-RDM
    sf_1rdm = sci.make_average_rdm(1)
    sf_1rdm_ci = ci.make_average_rdm(1)
    assert np.allclose(sf_1rdm, sf_1rdm_ci, atol=1e-8)

    # Test the 2-RDM
    sf_2rdm = sci.make_average_rdm(2)
    sf_2rdm_ci = ci.make_average_rdm(2)
    assert np.allclose(sf_2rdm, sf_2rdm_ci, atol=1e-8)

    # Test the 2-cumulant, per-root and state-averaged
    assert np.allclose(
        sci.make_cumulant(0, order=2, kind="sf"),
        ci.make_cumulant(0, order=2, kind="sf"),
        atol=1e-8,
    )
    assert np.allclose(
        sci.make_average_cumulant(2), ci.make_average_cumulant(2), atol=1e-8
    )

    # Selected CI has no 3-RDM, so it offers no 3-cumulant either
    with pytest.raises(ValueError, match=r"order must be one of \(2,\)"):
        sci.make_cumulant(0, order=3, kind="sf")


def _lih_noncontiguous_mo_space(system):
    return MOSpace(
        nmo=system.nmo,
        active_orbitals=[0, 1, 2, 3],
        frozen_virtual_orbitals=[4],
    )


def _lih_rhf():
    xyz = "Li 0.0 0.0 0.0\nH  0.0 0.0 3.0"
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    return RHF(charge=0, e_tol=1e-12)(system)


def test_sci_semicanonical_noncontiguous_mo_space():
    """final_orbitals='semicanonical' with a non-contiguous active space.

    active_orbitals=[0, 1, 2, 3] is followed by a frozen virtual at original
    index 4, which sits *before* the regular virtual in the original
    ordering but *after* it in the contiguous [active, virt, frozen_virt]
    ordering used internally, so mo_space.contig_to_orig is a genuine,
    non-trivial permutation.
    """
    rhf = _lih_rhf()
    mo_space = _lih_noncontiguous_mo_space(rhf.system)
    sci_params = SelectedCIParams(var_threshold=1e-10, pt2_threshold=1e-12)

    sci_original = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        mo_space_override=mo_space,
        sci_params=sci_params,
    )(rhf)
    sci_original.run()
    sci_semicanonical = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        mo_space_override=mo_space,
        sci_params=sci_params,
        final_orbitals="semicanonical",
    )(rhf)
    sci_semicanonical.run()

    np.testing.assert_array_equal(mo_space.orig_to_contig, [0, 1, 2, 3, 5, 4])
    np.testing.assert_array_equal(mo_space.contig_to_orig, [0, 1, 2, 3, 5, 4])
    assert sci_semicanonical.E[0] == approx(sci_original.E[0])


def test_sci_natural_noncontiguous_mo_space():
    """final_orbitals='natural' with the same non-contiguous active space as
    test_sci_semicanonical_noncontiguous_mo_space."""
    rhf = _lih_rhf()
    mo_space = _lih_noncontiguous_mo_space(rhf.system)
    sci_params = SelectedCIParams(var_threshold=1e-10, pt2_threshold=1e-12)

    sci_original = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        mo_space_override=mo_space,
        sci_params=sci_params,
    )(rhf)
    sci_original.run()
    sci_natural = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        mo_space_override=mo_space,
        sci_params=sci_params,
        final_orbitals="natural",
    )(rhf)
    sci_natural.run()

    assert sci_natural.E[0] == approx(sci_original.E[0])

    g1_act = sci_natural.make_average_rdm(1)
    off_diag = g1_act - np.diag(np.diag(g1_act))
    assert np.max(np.abs(off_diag)) < 1e-6


@pytest.mark.slow
def test_sci_water_core_excited():
    """Test SelectedCI on a water core-excited state."""
    # This should be converged to the following GASCI input
    # from forte2.ci import CI
    # ci = CI(
    #     states=State(nel=10, multiplicity=1, ms=0.0, gas_max=[1], gas_min=[1]),
    #     active_orbitals=[[0], list(range(1, 13))],
    #     nroots=3,
    #     davidson_liu_params=DavidsonLiuParams(
    #         e_tol=1e-10,
    #         r_tol=1e-5,
    #     ),
    # )(rhf)
    # ci.run()

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(xyz=xyz, basis_set="6-31g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0)(system)

    ci_strings = CIStrings(5, 5, 0, [[0], [0], [0] * 11], [1, 2], [1, 2])
    guess_dets = ci_strings.make_determinants()

    ci = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(13)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=1e-10,
            guess_dets=guess_dets,
            do_spin_penalty=True,
            screening_criterion="hbci",
            # do not allow the core orbital occupation to change from the guess determinants
            frozen_annihilation=[0],
            frozen_creation=[0],
            num_batches_per_thread=16,
        ),
        davidson_liu_params=DavidsonLiuParams(
            e_tol=1e-10,
            r_tol=1e-5,
        ),
    )(rhf)
    ci.run()

    assert ci.E[0] == pytest.approx(-56.34437851987155, abs=1e-6)


@pytest.mark.slow
def test_sci_water_core_excited_with_gasscf_orbs():
    """Test SelectedCI on a water core-excited state."""
    # This should be converged to the following GASCI input
    # from forte2 import CISolver, MCOptimizer, CI
    # xyz = """
    # O   0.0000000000  -0.0000000000  -0.0662628033
    # H   0.0000000000  -0.7540256101   0.5259060578
    # H  -0.0000000000   0.7530256101   0.5260060578
    # """

    # system = System(xyz=xyz, basis_set="6-31g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    # rhf = RHF(charge=0)(system)

    # ci = CISolver(
    #     states=State(nel=10, multiplicity=1, ms=0.0, gas_max=[1, 2], gas_min=[1, 2]),
    #     active_orbitals=[[0], [1], list(range(2, 7))],
    #     davidson_liu_params=DavidsonLiuParams(
    #         e_tol=1e-10,
    #         r_tol=1e-5,
    #     ),
    # )
    # mc = MCOptimizer(ci)(rhf)
    # mc.run()

    # ci = CI(
    #     states=State(nel=10, multiplicity=1, ms=0.0, gas_max=[1], gas_min=[1]),
    #     active_orbitals=[[0], list(range(1, 13))],
    #     davidson_liu_params=DavidsonLiuParams(
    #         e_tol=1e-10,
    #         r_tol=1e-5,
    #     ),
    # )(mc)
    # ci.run()

    from forte2 import CISolver, MCOptimizer

    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(xyz=xyz, basis_set="6-31g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0)(system)

    # Small active space GASSCF to get good initial orbitals for sCI
    ci = CISolver(
        states=State(nel=10, multiplicity=1, ms=0.0, gas_max=[1, 2], gas_min=[1, 2]),
        active_orbitals=[[0], [1], list(range(2, 7))],
        davidson_liu_params=DavidsonLiuParams(
            e_tol=1e-10,
            r_tol=1e-5,
        ),
    )
    mc = MCOptimizer(ci)(rhf)
    mc.run()

    guess_dets = ci.sub_solvers[0].ci_strings.make_determinants()

    ci = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(13)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=1e-10,
            guess_dets=guess_dets,
            do_spin_penalty=True,
            screening_criterion="hbci",
            # do not allow the core orbital occupation to change from the guess determinants
            frozen_annihilation=[0],
            frozen_creation=[0],
            num_batches_per_thread=16,
        ),
        davidson_liu_params=DavidsonLiuParams(
            e_tol=1e-10,
            r_tol=1e-5,
        ),
    )(mc)
    ci.run()

    assert ci.E[0] == pytest.approx(-56.3574249874, abs=1e-6)


def test_sci_n2_multiple_roots():
    """Test that multiple roots can be converged for N2."""
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.1
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0)(system)
    sci = SelectedCI(
        states=State(nel=14, multiplicity=1, ms=0.0),
        active_orbitals=list(range(10)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-8,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            screening_criterion="hbci",
            guess_occ_window=3,
            guess_vir_window=3,
            num_batches_per_thread=16,
        ),
        die_if_not_converged=False,
        nroots=2,
        davidson_liu_params=DavidsonLiuParams(
            e_tol=1e-10,
            r_tol=1e-5,
            ndets_per_guess=20,
        ),
    )(rhf)
    sci.run()

    assert sci.E[0] == pytest.approx(-108.70183536777276, abs=1e-8)
    assert sci.E[1] == pytest.approx(-108.35946592810289, abs=1e-8)


def test_sci_water_valence_excitation():
    """Test SelectedCI on a water valence-excited state."""
    xyz = """
    O   0.0000000000  -0.0000000000  -0.0662628033
    H   0.0000000000  -0.7540256101   0.5259060578
    H  -0.0000000000   0.7530256101   0.5260060578
    """

    system = System(xyz=xyz, basis_set="6-31g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0)(system)

    sci = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(13)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=3e-4,
            pt2_threshold=1e-8,
            do_spin_penalty=True,
            screening_criterion="hbci",
            guess_occ_window=3,
            guess_vir_window=1,
            num_batches_per_thread=16,
        ),
        die_if_not_converged=False,
        nroots=2,
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        do_test_rdms=True,
    )(rhf)
    sci.run()
    assert sci.E[0] == pytest.approx(-76.12037086, abs=1e-6)
    assert sci.E[1] == pytest.approx(-75.80852593, abs=1e-6)


@pytest.mark.slow
def test_sci_1tdm_water_core_excited_and_gs():
    """Test the 1-TDM between a water core-excited state and the ground state."""
    xyz = """
    O            0.000000000000     0.000000000000    -0.069592187400
    H            0.000000000000    -0.783151105291     0.552239257834
    H            0.000000000000     0.783151105291     0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="6-31g", auxiliary_basis_set="def2-universal-jkfit"
    )
    rhf = RHF(charge=0)(system)

    gs_params = SelectedCIParams(
        var_threshold=1e-4,
        pt2_threshold=1e-8,
        guess_occ_window=3,
        guess_vir_window=1,
        num_batches_per_thread=16,
    )
    gs_ci_params = DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5)

    # the singly core-excited determinants make the largest contributions
    # to the final state, so we can use a large set here
    ci_strings_s = CIStrings(5, 5, 0, [[0], [0], [0] * 11], [1, 2], [1, 2])
    gas_guess_dets = ci_strings_s.make_determinants()
    # if the doubly core-excited determinants are included together with the
    # singles, they will almost certainly be filtered out by the initial guess routine
    # due to them being much higher in energy.
    # So we need to pin a small set of them to ensure they are included in the variational space.
    # This is necessary because orbital 0 is frozen, so if they're not included in the initial guess,
    # they will never be generated.
    # Allowing annihilation in the 0-th orbital is theoretically able to overcome this,
    # but the interals involved are sufficiently small that, unless a very tight threshold is used,
    # the doubles will alwayus be rejected in the selection process.
    pinned_guess_dets = [
        Determinant("022222"),
        Determinant("02222ab"),
        Determinant("02222ba"),
    ]
    gas_params = gs_params.copy(
        guess_dets=gas_guess_dets,
        frozen_annihilation=[0],
        frozen_creation=[0],
        pinned_guess_dets=pinned_guess_dets,
    )
    # the defaul ndets_per_guess is too small, causing convergence onto a higher root.
    gas_ci_params = gs_ci_params.copy(ndets_per_guess=50)

    sci = SelectedCI(
        states=[
            State(nel=10, multiplicity=1, ms=0.0),
            State(nel=10, multiplicity=1, ms=0.0),
        ],
        active_orbitals=list(range(13)),
        sci_params=[gas_params, gs_params],
        davidson_liu_params=[gas_ci_params, gs_ci_params],
        nroots=[1, 1],
        do_transition_dipole=True,
    )(rhf)
    sci.run()

    assert sci.E[0] == pytest.approx(-56.36320838, abs=1e-8)
    assert sci.E[1] == pytest.approx(-76.12086372, abs=1e-8)
    assert sci.oscillator_strengths[(1, 0)] == pytest.approx(0.020634, abs=1e-4)


def _lih_rhf_tight():
    """Small LiH reference (tight SCF) shared by the regression tests below."""
    system = System(
        xyz="Li 0.0 0.0 0.0\nH  0.0 0.0 3.0",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    return system, RHF(charge=0, e_tol=1e-10)(system)


def test_sci_per_state_params_are_not_shared():
    """A single SelectedCIParams shared by several states must be copied per state.

    ``_initial_guess`` rebinds ``sci_params.guess_dets``; when one object was
    aliased across states, state 1 inherited state 0's guess determinants and
    validated them against its own electron count. With states of differing
    alpha/beta counts that raised
    "Guess determinant ... has 1 alpha electrons, expected 2".
    """
    _, rhf = _lih_rhf_tight()
    sci = SelectedCI(
        states=[
            State(nel=4, multiplicity=1, ms=0.0),  # active (na, nb) = (1, 1)
            State(nel=4, multiplicity=3, ms=1.0),  # active (na, nb) = (2, 0)
        ],
        nroots=[1, 1],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5],
        sci_params=SelectedCIParams(guess_occ_window=1),
    )(rhf)
    sci.run()

    # each sub-solver must own its params, and its own guess determinants
    assert sci.sub_solvers[0].sci_params is not sci.sub_solvers[1].sci_params
    assert sci.sub_solvers[0].davidson_liu_params is not (
        sci.sub_solvers[1].davidson_liu_params
    )
    for solver, (na, nb) in zip(sci.sub_solvers, [(1, 1), (2, 0)]):
        for d in solver.sci_params.guess_dets:
            assert (d.count_alpha(), d.count_beta()) == (na, nb)


def test_sci_set_ints_then_run_updates_slater_rules():
    """set_ints followed by run() must push the new integrals into the sci helper's internal
    SlaterRules -- there is no separate, persistent Python-level SlaterRules to keep in sync
    anymore; see SelectedCIHelper.slater_rules."""
    _, rhf = _lih_rhf_tight()
    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5],
        sci_params=SelectedCIParams(guess_occ_window=1),
    )(rhf)
    sci.run()

    worker = sci.sub_solvers[0]
    sci_helper_before = worker.sci_helper
    dets = worker.sci_helper.dets()
    before = worker.sci_helper.slater_rules(dets, 0, 0)

    # shifting only the scalar term must shift every diagonal element by the same amount
    sci.set_ints(worker.ints.E + 1.0, worker.ints.H, worker.ints.V)
    sci.run()

    assert worker.sci_helper is sci_helper_before
    dets = worker.sci_helper.dets()
    after = worker.sci_helper.slater_rules(dets, 0, 0)
    assert after - before == approx(1.0)


@pytest.mark.parametrize("final_orbitals", ["original", "semicanonical", "natural"])
def test_sci_final_orbitals(final_orbitals):
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
        final_orbitals=final_orbitals,
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(-2.180967812920)


def test_sci_pt2_split_is_independent_of_var_threshold():
    """
    ept2_var + ept2_pt is the Epstein-Nesbet PT2 over the determinants outside the
    variational space, so it cannot depend on var_threshold. That threshold only decides how the total is split between the two reported energies.
    """
    pt2_threshold = 1e-10
    xyz = """
        O 0.0000000 0.0000000 -0.0655905
        H 0.0000000 -0.7578344 0.5203775
        H 0.0000000  0.7578344 0.5203775
        """
    system = System(xyz=xyz, basis_set="6-31g", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    sci = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(13)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-3,
            pt2_threshold=pt2_threshold,
            maxcycle=2,
        ),
    )(rhf)
    sci.run()

    # Freeze the variational space and its coefficients
    # then run selection at different eps_var but fixed eps_pt2
    worker = sci.sub_solvers[0]
    dets = worker.dets
    c = worker.evecs
    energies = np.ascontiguousarray(np.asarray(worker.evals).real)

    def probe(var_threshold):
        helper = SelectedCIHelper(
            worker.norb, dets, c, worker.ints.E, worker.ints.H, worker.ints.V, 0
        )
        helper.set_energies(energies)
        helper.select_hbci(var_threshold=var_threshold, pt2_threshold=pt2_threshold)
        return np.array(helper.ept2_var()), np.array(helper.ept2_pt())

    # very large eps_var leaves all to PT2 space
    # all subsequent e_var + e_pt2 must add to ref_pt
    ref_var, ref_pt = probe(1e6)
    assert np.all(ref_var == 0.0)

    for var_threshold in (1e-3, 1e-4, 1e-5, 1e-7, 0.0):
        var, pt = probe(var_threshold)
        assert var + pt == approx(ref_pt)
