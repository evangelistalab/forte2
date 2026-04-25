import numpy as np
import pytest

from forte2 import System, State, Determinant, CIStrings
from forte2.ci import CI
from forte2.scf import RHF
from forte2.sci import SelectedCI
from forte2._forte2 import SelectedCIHelper
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
            num_threads=4,
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
            num_threads=4,
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
            num_threads=4,
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
            num_threads=4,
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

    sci0 = SelectedCI(
        states=State(nel=10, multiplicity=1, ms=0.0),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            num_threads=4,
            num_batches_per_thread=16,
        ),
        nroots=1,
    )(rhf)

    sci0.run()

    sci = SelectedCI(
        states=State(nel=9, multiplicity=2, ms=0.5),
        active_orbitals=list(range(12)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-5,
            pt2_threshold=0.0,
            do_spin_penalty=True,
            guess_dets=[Determinant("a2222")],
            num_threads=4,
            num_batches_per_thread=16,
        ),
        nroots=1,
    )(rhf)

    sci.run()

    # This value is sensitive to the selected space growth details; keep a practical tolerance.
    assert sci.E[0] == pytest.approx(-96.5578779686, abs=5e-3)


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
            num_threads=4,
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

    rdm1 = sci.sub_solvers[0].make_sf_1rdm(0)
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


def test_sci_1trdm_validates_helper_compatibility():
    """SelectedCI transition 1-RDMs validate helper compatibility."""
    h2 = np.zeros((2, 2))
    v2 = np.zeros((2, 2, 2, 2))
    h3 = np.zeros((3, 3))
    v3 = np.zeros((3, 3, 3, 3))

    helper = SelectedCIHelper(
        2, [Determinant("20")], np.array([[1.0]]), 0.0, h2, v2, 0
    )
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
            np.abs(ci.transition_dipoles[key]), abs=1e-5
        )
        assert sci.oscillator_strengths[key] == pytest.approx(
            ci.oscillator_strengths[key], abs=1e-5
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
        assert np.allclose(sci.make_sf_1rdm(*key), ci.make_sf_1rdm(*key), atol=1e-6)

    for key in ci.transition_dipoles:
        assert np.abs(sci.transition_dipoles[key]) == pytest.approx(
            np.abs(ci.transition_dipoles[key]), abs=1e-5
        )
        assert sci.oscillator_strengths[key] == pytest.approx(
            ci.oscillator_strengths[key], abs=1e-5
        )


def test_sci_claude_algorithms_flag():
    """Claude sigma kernels should be opt-in and preserve the SCI energy."""
    rhf = _h4_rhf()

    sci_default = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            use_claude_algorithms=False,
        ),
    )(rhf)
    sci_default.run()

    sci_claude = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            use_claude_algorithms=True,
        ),
    )(rhf)
    sci_claude.run()

    assert sci_default.E[0] == approx(-2.180967812920)
    assert sci_claude.E[0] == approx(sci_default.E[0])


def test_sci_make_rdms():
    """Test 1- and 2-RDMs from SelectedCI."""
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
    sf_1rdm = sci.make_average_1rdm()
    sf_1rdm_ci = ci.make_average_1rdm()
    assert np.allclose(sf_1rdm, sf_1rdm_ci, atol=1e-8)

    # Test the 2-RDM
    sf_2rdm = sci.make_average_2rdm()
    sf_2rdm_ci = ci.make_average_2rdm()
    assert np.allclose(sf_2rdm, sf_2rdm_ci, atol=1e-8)


def test_sci_semicanonical_final_orbital():
    """Semicanonical final orbital path should execute without runtime errors."""
    rhf = _h4_rhf()

    sci = SelectedCI(
        states=State(nel=4, multiplicity=1, ms=0.0),
        active_orbitals=list(range(4)),
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
        final_orbital="semicanonical",
    )(rhf)
    sci.run()

    assert sci.E[0] == approx(-2.180967812920)


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

    ci_strings = CIStrings(5, 5, 0, [[0], [0], [0] * 11], [0, 2], [1, 2])
    # this has FCI length, but will be wittled down by the HBCI initial guess
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
            num_threads=4,
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
            num_threads=4,
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
            num_threads=4,
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
            num_threads=4,
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
