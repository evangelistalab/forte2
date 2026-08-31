import numpy as np
import pytest

from forte2 import System, GHF, SpinorUpcaster, MCOptimizer, MOSpace, X2CParams
from forte2.ci import RelCI
from forte2.sci import RelSelectedCI, RelSelectedCISolver
from forte2.helpers.comparisons import approx
from forte2.base_classes.params import SelectedCIParams
from forte2.lib.det import Determinant
from forte2.lib.ci_helpers import RelSelectedCIHelper

from rdm_debug_utils import make_so_1rdm_debug, make_so_2rdm_debug


def _h2_ghf_upcast():
    """Smallest 2c case: H2/STO-6G -> 4 spinors, 2 electrons (6 determinants)."""
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    # a single shared upcaster instance so that RDM-basis-dependent comparisons
    # use identical spinor orbitals (apply_random_phase would otherwise differ)
    return SpinorUpcaster(apply_random_phase=True)(scf)


def test_rel_sci_h2():
    """RelSelectedCI at tight thresholds == 2c-FCI energy of H2 (test_rel_ci_h2)."""
    conv = _h2_ghf_upcast()

    sci = RelSelectedCI(
        nel=2,
        active_orbitals=4,
        do_test_rdms=True,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )(conv)
    sci.run()

    assert sci.E[0] == approx(-1.096071975854)
    # the full variational space is recovered, so there is no PT2 remainder
    assert abs(sci.E_pt2[0]) < 1e-10


def test_rel_sci_pinned_only_guess():
    """RelSelectedCI must support an initial guess built purely from pinned_guess_dets
    (test_sci_pinned_only_guess's counterpart): _initial_guess previously left
    guess_hdiag/nguess_dets unbound when guess_dets was empty."""
    conv = _h2_ghf_upcast()

    sci = RelSelectedCI(
        nel=2,
        active_orbitals=4,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            guess_dets=[],
            pinned_guess_dets=[Determinant("aa00")],
        ),
    )(conv)
    sci.run()

    assert sci.E[0] == approx(-1.096071975854)


def test_rel_sci_h2_exact():
    """Exact selected-CI diagonalization path for the 2c (complex) case."""
    conv = _h2_ghf_upcast()

    sci = RelSelectedCI(
        nel=2,
        active_orbitals=4,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
            ci_algorithm="exact",
        ),
    )(conv)
    sci.run()

    assert sci.E[0] == approx(-1.096071975854)


def test_rel_sci_h2_rdms_match_rel_ci():
    """Complex spin-orbital 1-/2-RDMs match RelCI in the full space.

    RDMs are basis dependent, so RelCI and RelSelectedCI are built on the *same*
    spinor orbitals (one shared upcaster) before comparing.
    """
    conv = _h2_ghf_upcast()

    ci = RelCI(nel=2, active_orbitals=4)(conv)
    ci.run()

    sci = RelSelectedCI(
        nel=2,
        active_orbitals=4,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )(conv)
    sci.run()

    assert sci.E[0] == approx(ci.E[0])
    assert np.allclose(sci.make_average_rdm(1), ci.make_average_rdm(1), atol=1e-8)
    assert np.allclose(sci.make_average_rdm(2), ci.make_average_rdm(2), atol=1e-8)

    # the 2-cumulant, per-root and state-averaged
    assert np.allclose(
        sci.make_cumulant(0, order=2, kind="so"),
        ci.make_cumulant(0, order=2, kind="so"),
        atol=1e-8,
    )
    assert np.allclose(
        sci.make_average_cumulant(2), ci.make_average_cumulant(2), atol=1e-8
    )

    # Selected CI has no 3-RDM, so it offers no 3-cumulant either
    with pytest.raises(ValueError, match=r"order must be one of \(2,\)"):
        sci.make_cumulant(0, order=3, kind="so")


def test_rel_sci_hf_ghf():
    """Spin-orbit X2C 2c-FCI energy of HF (test_rel_ci_hf_ghf).

    This is the genuinely complex (spin-orbit) correctness check: the one- and
    two-electron integrals are complex Hermitian, not merely a real problem
    promoted to complex.
    """
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )
    scf = GHF(charge=0)(system)

    sci = RelSelectedCI(
        nel=10,
        core_orbitals=2,
        active_orbitals=12,
        do_test_rdms=True,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )(scf)
    sci.run()

    assert sci.E[0] == approx(-100.10065023157668)


def test_rel_sci_hf_ghf_transition_rdms():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )
    scf = GHF(charge=0)(system)

    nroots = 3
    sci = RelSelectedCI(
        nel=10,
        nroots=nroots,
        core_orbitals=2,
        active_orbitals=12,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )(scf)
    sci.run()

    # sub_solver holds the per-root vectors and the SparseState reference RDMs
    ref_solver = sci.sub_solvers[0]
    nel_active = 8  # 10 electrons - 2 core spinors

    for left in range(nroots):
        for right in range(nroots):
            g1 = sci.make_rdm(left, right, order=1, kind="so")
            g1_ref = make_so_1rdm_debug(ref_solver, left, right)
            assert np.allclose(g1, g1_ref, atol=1e-10)
            # Hermiticity of the (transition) 1-RDM: gamma(l,r) = gamma(r,l)^dagger
            assert np.allclose(
                g1,
                sci.make_rdm(right, left, order=1, kind="so").conj().T,
                atol=1e-10,
            )
            # Tr[gamma1(l,r)] = <l|N|r> = nel_active * delta_lr
            assert np.trace(g1) == approx(nel_active if left == right else 0.0)

    # 2-RDM: check a diagonal root and an off-diagonal transition pair against the
    # reference, plus the transition-RDM Hermiticity convention.
    for left, right in [(0, 0), (0, 1)]:
        g2 = sci.make_rdm(left, right, order=2, kind="so")
        g2_ref = make_so_2rdm_debug(ref_solver, left, right)
        assert np.allclose(g2, g2_ref, atol=1e-10)
        # gamma2(l,r)[p,q,r,s] = <l|a+_p a+_q a_s a_r|r> => gamma2(l,r) = conj(gamma2(r,l)^T_(2,3,0,1))
        g2_swapped = (
            sci.make_rdm(right, left, order=2, kind="so").conj().transpose(2, 3, 0, 1)
        )
        assert np.allclose(g2, g2_swapped, atol=1e-10)


def test_rel_sci_natural_noncontiguous_mo_space():
    """
    RelSelectedCI final_orbitals='natural' reproduces the energy and
    natural occupation number spectrum after a non-trivial
    contig/orig permutation
    """
    xyz = """
    Li 0.0 0.0 0.0
    H  0.0 0.0 3.0
    """
    system = System(
        xyz=xyz,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    scf = GHF(charge=0, e_tol=1e-12)(system)
    mo_space = MOSpace(
        nmo=system.nmo * 2,
        core_orbitals=[0, 1],
        active_orbitals=[2, 3, 4, 5],
        frozen_virtual_orbitals=[6, 7],
    )
    sci_params = SelectedCIParams(
        selection_algorithm="hbci",
        var_threshold=1e-12,
        pt2_threshold=0.0,
    )

    sci_original = RelSelectedCI(
        nel=4, mo_space_override=mo_space, sci_params=sci_params
    )(scf)
    sci_original.run()
    sci_natural = RelSelectedCI(
        nel=4,
        mo_space_override=mo_space,
        sci_params=sci_params,
        final_orbitals="natural",
    )(scf)
    sci_natural.run()

    assert sci_natural.E[0] == approx(sci_original.E[0])
    np.testing.assert_allclose(
        sci_natural.mos.C[0].conj().T @ system.ints_overlap() @ sci_natural.mos.C[0],
        np.eye(mo_space.nmo),
        atol=1e-10,
    )

    original_occs = np.sort(np.linalg.eigvalsh(sci_original.make_average_rdm(1)))[::-1]
    natural_occs = np.sort(np.linalg.eigvalsh(sci_natural.make_average_rdm(1)))[::-1]
    assert natural_occs == approx(original_occs)


@pytest.mark.slow
def test_rel_sci_casscf_hf_ghf():
    """RelSelectedCISolver drops into relativistic CASSCF"""
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )
    scf = GHF(charge=0)(system)

    ci_solver = RelSelectedCISolver(
        nel=10,
        nroots=1,
        core_orbitals=2,
        active_orbitals=12,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )
    mc = MCOptimizer(ci_solver)(scf)
    mc.run()

    assert mc.E == approx(-100.1361832608)


def test_rel_sci_transition_dipole():
    conv = _h2_ghf_upcast()

    sci = RelSelectedCI(
        nel=2,
        active_orbitals=4,
        nroots=2,
        do_transition_dipole=True,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
    )(conv)
    sci.run()

    # no bright transitions, just an existence check
    assert sci.transition_dipoles[(0, 0)].shape == (3,)
    assert sci.oscillator_strengths[(0, 0)] == 0.0
    assert sci.vertical_transition_energies[(0, 0)] == 0.0
    assert sci.vertical_transition_energies[(0, 1)] == approx(sci.E[1] - sci.E[0])


@pytest.mark.parametrize("final_orbitals", ["original", "semicanonical", "natural"])
def test_rel_sci_final_orbitals(final_orbitals):
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c=X2CParams(x2c_type="so"),
    )
    scf = GHF(charge=0)(system)

    sci = RelSelectedCI(
        nel=10,
        core_orbitals=2,
        active_orbitals=12,
        do_test_rdms=True,
        sci_params=SelectedCIParams(
            selection_algorithm="hbci",
            var_threshold=1e-12,
            pt2_threshold=0.0,
        ),
        final_orbitals=final_orbitals,
    )(scf)
    sci.run()

    assert sci.E[0] == approx(-100.10065023157668)


def test_rel_sci_pt2_split_is_independent_of_var_threshold():
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
    rhf = GHF(charge=0, e_tol=1e-12)(system)

    sci = RelSelectedCI(
        nel=10,
        active_orbitals=list(range(26)),
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
        helper = RelSelectedCIHelper(
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
