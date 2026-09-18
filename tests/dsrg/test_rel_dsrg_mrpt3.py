import numpy as np
import pytest

from forte2 import System, GHF, CI, MCOptimizer, RelCISolver, X2CParams, AVAS
from forte2.dsrg import RelDSRG_MRPT2, RelDSRG_MRPT3, RelFNO_DSRG_MRPT3
from forte2.helpers.comparisons import approx
from forte2.data.atom_data import EH_TO_WN


def test_mrpt3_n2_nonrel():
    erhf = -108.954140898736
    emcscf = -109.0811491968

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = GHF(charge=0)(system)
    rhf.run()
    rng = np.random.default_rng(1234)
    random_phase = np.diag(np.exp(1j * rng.uniform(-np.pi, np.pi, size=rhf.nmo * 2)))
    rhf.C[0] = rhf.C[0] @ random_phase

    ci_solver = RelCISolver(
        nel=14,
        core_orbitals=8,
        active_orbitals=12,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    assert rhf.E == approx(erhf)
    assert mc.E == approx(emcscf)

    dsrg = RelDSRG_MRPT3(
        frozen_core_orbitals=4,
        flow_param=0.5,
        relax_reference="iterate",
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0] == approx(
        [-109.25301485009223, -109.2538362628393, -109.08114919682387]
    )
    assert dsrg.relax_energies[1] == approx(
        [-109.25344887585058, -109.25344888535047, -109.0802678007682]
    )
    assert dsrg.relax_energies[2] == approx(
        [-109.25344824472272, -109.25344824472299, -109.08026606599341]
    )


def test_mrpt3_f_atom_rel_sa():
    xyz = """
    F 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVTZ",
        auxiliary_basis_set="cc-pVQZ-JKFIT",
        x2c=X2CParams(x2c_type="so", snso_type="row-dependent"),
        use_gaussian_charges=True,
    )
    mf = GHF(charge=-1, die_if_not_converged=False)(system)
    ci_solver = RelCISolver(
        nel=9,
        nroots=6,
        active_orbitals=8,
        core_orbitals=2,
    )
    mc = MCOptimizer(
        ci_solver,
        e_tol=1e-8,
        g_tol=1e-6,
    )(mf)
    dsrg = RelDSRG_MRPT3(flow_param=0.35, relax_reference="once")(mc)
    dsrg.run()
    assert (dsrg.relax_eigvals[4] - dsrg.relax_eigvals[3]) * EH_TO_WN == pytest.approx(
        400.1722015310902, abs=1e-2
    )


def test_mrpt3_fno():
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = GHF(charge=0)(system)
    rhf.run()
    rng = np.random.default_rng(1234)
    random_phase = np.diag(np.exp(1j * rng.uniform(-np.pi, np.pi, size=rhf.nmo * 2)))
    rhf.C[0] = rhf.C[0] @ random_phase

    ci_solver = RelCISolver(
        nel=14,
        core_orbitals=8,
        active_orbitals=12,
    )
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    pt3_ref = RelDSRG_MRPT3(flow_param=0.5, frozen_core_orbitals=4)(mc)
    pt3_ref.run()

    # p_o=1.0 retains every virtual orbital, so the truncation correction
    # should vanish identically and the FNO-corrected PT3 energy must
    # reproduce the untruncated result to near machine precision -- a real
    # correctness check on the composition, not an accuracy statement about
    # truncation.
    fno_100 = RelFNO_DSRG_MRPT3(
        flow_param=0.5, fno_flow_param=0.5, fno_p_o=1.0, frozen_core_orbitals=4
    )(mc)
    fno_100.run()
    shift = fno_100.pt2_fno.hbar_shift
    assert shift["e_dsrg"] == approx(0.0)
    assert shift["hbar0"] == approx(0.0)
    assert shift["hbar1"] == approx(np.zeros_like(shift["hbar1"]))
    assert shift["hbar2"] == approx(np.zeros_like(shift["hbar2"]))
    assert fno_100.E_dsrg == approx(pt3_ref.E_dsrg)

    # a genuinely truncated case: the correction is non-zero and brings the
    # FNO-space PT3 energy within a bounded, physically sane distance of the
    # untruncated reference for this small basis.
    fno = RelFNO_DSRG_MRPT3(
        flow_param=0.5, fno_flow_param=0.5, fno_p_o=0.9, frozen_core_orbitals=4
    )(mc)
    fno.run()
    assert fno.pt2_full.mo_space.nvirt < fno_100.pt2_full.mo_space.nvirt
    assert abs(fno.pt2_fno.hbar_shift["e_dsrg"]) > 1e-6
    assert abs(fno.E_dsrg - pt3_ref.E_dsrg) < 0.05

    # with reference relaxation, the correction is folded into the effective
    # Hamiltonian handed to the CI solver (not applied as a post-hoc shift),
    # so the relaxed (single, here) eigenvalue must stay consistent with the
    # state-averaged relaxed energy it comes from, and relaxation must lower
    # the energy relative to the fixed-reference result.
    fno_relaxed = RelFNO_DSRG_MRPT3(
        flow_param=0.5,
        fno_flow_param=0.5,
        fno_p_o=0.9,
        frozen_core_orbitals=4,
        relax_reference="once",
    )(mc)
    fno_relaxed.run()
    assert fno_relaxed.relax_eigvals[0] == approx(fno_relaxed.E_relaxed_ref)
    assert fno_relaxed.E_relaxed_ref.real < fno_relaxed.E_dsrg.real


def test_mrpt3_fno_vs_forte():
    """
    Cross-validation against forte1 input:
    https://gist.github.com/brianz98/57770edfe3b883f1dd7991f07ff9ee7e
    """
    e_pt2_forte = -109.2357480291
    e_pt3_forte = -109.2594557723
    e_pt3_relaxed_forte = -109.2598269214

    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = GHF(charge=0)(system)
    rhf.run()

    ci_solver = RelCISolver(nel=14, core_orbitals=8, active_orbitals=12)
    mc = MCOptimizer(ci_solver)(rhf)
    mc.run()

    # s1 builds the FNOs and the correction; the high-level method's own s2 is
    # independent (and deliberately different here, to check the two never get
    # conflated).
    s1, s2 = 1.5, 2.0

    fno = RelFNO_DSRG_MRPT3(
        flow_param=s2, fno_flow_param=s1, fno_p_o=0.9, frozen_core_orbitals=4
    )(mc)
    fno.run()

    # pt2_fno does not apply its own correction; a consumer does.
    pt2_fno = fno.pt2_fno
    assert (pt2_fno.E_dsrg + pt2_fno.hbar_shift["e_dsrg"]).real == approx(e_pt2_forte)
    assert fno.E_dsrg.real == approx(e_pt3_forte)

    fno_relaxed = RelFNO_DSRG_MRPT3(
        flow_param=s2,
        fno_flow_param=s1,
        fno_p_o=0.9,
        frozen_core_orbitals=4,
        relax_reference="once",
    )(mc)
    fno_relaxed.run()
    assert fno_relaxed.E_relaxed_ref.real == approx(e_pt3_relaxed_forte)


def test_mrpt3_ch_snso_kramers():
    """CH has an odd number of electrons, so its roots must be Kramers degenerate.

    The reference is a 2-Pi state, so the four roots form two pairs that are exactly
    degenerate by time-reversal symmetry. That makes this a zero-tolerance check
    needing no external reference. It only has power when the state-averaged 1-RDM is
    genuinely complex in the semicanonical basis, which is why the assertion on
    max|Im(gamma1)| is here: an atomic reference has a real 1-RDM and would let a
    convention error in the active mean field pass unnoticed.
    """
    escf = -38.286254865078
    emcscf = -38.320115307064
    # The two DSRG energies moved by 4.3e-7 when H1_T2_C1_non_od was added to
    # _RelDSRGHelper; the Kramers and spin-orbit assertions below did not.
    edsrg = -38.433775292986
    erelaxed = -38.434549655544
    soc_wn = 27.424588

    xyz = """
    C 0.0 0.0 0.0
    H 0.0 0.0 1.1199
    """

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        minao_basis_set="ano-r0",
        x2c=X2CParams(x2c_type="so", snso_type="row-dependent"),
        use_gaussian_charges=True,
    )
    mf = GHF(charge=0, die_if_not_converged=False)(system)
    avas = AVAS(
        subspace=["H(1s)", "C(2s)", "C(2p)"],
        selection_method="separate",
        num_active_docc=5,
        num_active_uocc=5,
    )(mf)
    mc = MCOptimizer(
        RelCISolver(nel=7, nroots=4), e_tol=1e-9, g_tol=1e-7, die_if_not_converged=False
    )(avas)
    mc.run()

    assert mf.E == approx(escf)
    assert mc.E.real == approx(emcscf)
    # Without a complex 1-RDM this test cannot see a conjugation error.
    assert np.abs(np.imag(mc.make_average_rdm(1))).max() > 1e-4

    dsrg = RelDSRG_MRPT3(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.E_dsrg.real == approx(edsrg)
    assert dsrg.E_relaxed_ref.real == approx(erelaxed)

    levels = np.sort(np.asarray(dsrg.relax_eigvals).ravel().real)
    levels = (levels - levels[0]) * EH_TO_WN
    kramers = [levels[1] - levels[0], levels[3] - levels[2]]
    # Conjugating gamma1 in the active mean field splits these by ~16 cm^-1.
    assert max(kramers) < 1e-2, f"Kramers pairs split by {max(kramers):.4f} cm^-1"
    assert levels[2] == pytest.approx(soc_wn, abs=1e-2)


def test_mrpt3_skip_3_cumulant_leaves_splittings_alone():
    """skip_3_cumulant must not move a single splitting.

    lambda3 reaches DSRG-MRPT3 only through the energy kernel, so it lands in the
    scalar part of Hbar and shifts every root by the same amount. The splittings are
    therefore untouched and E_dsrg simply loses that term. CH is used because its
    1-RDM is genuinely complex, so the reference is not accidentally simple.

    The reference is rebuilt for each variant on purpose: reference relaxation calls
    set_ints on the shared parent solver, so a second DSRG run against the same
    MCOptimizer would start from the first run's Hbar rather than the CASSCF
    reference.
    """
    xyz = """
    C 0.0 0.0 0.0
    H 0.0 0.0 1.1199
    """

    def run(skip):
        system = System(
            xyz=xyz,
            basis_set="decon-cc-pVDZ",
            auxiliary_basis_set="cc-pVTZ-JKFIT",
            minao_basis_set="ano-r0",
            x2c=X2CParams(x2c_type="so", snso_type="row-dependent"),
            use_gaussian_charges=True,
        )
        mf = GHF(charge=0, die_if_not_converged=False)(system)
        avas = AVAS(
            subspace=["H(1s)", "C(2s)", "C(2p)"],
            selection_method="separate",
            num_active_docc=5,
            num_active_uocc=5,
        )(mf)
        mc = MCOptimizer(
            RelCISolver(nel=7, nroots=4),
            e_tol=1e-9,
            g_tol=1e-7,
            die_if_not_converged=False,
        )(avas)
        mc.run()
        dsrg = RelDSRG_MRPT3(
            flow_param=0.5, relax_reference="once", skip_3_cumulant=skip
        )(mc)
        dsrg.run()
        e = np.sort(np.asarray(dsrg.relax_eigvals).ravel().real)
        return (e - e.min()) * EH_TO_WN, dsrg.E_dsrg.real

    levels_full, e_full = run(False)
    levels_skip, e_skip = run(True)

    assert np.abs(levels_skip - levels_full).max() < 1e-6
    # The term is real and non-negligible, so the test cannot pass by doing nothing.
    assert abs(e_skip - e_full) > 1e-4


def test_make_average_cumulants_max_order():
    """max_order=2 skips lambda3 entirely, and only 2 or 3 are accepted."""
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )
    rhf = GHF(charge=0)(system)
    rhf.run()
    ci = CI(RelCISolver(nel=14, core_orbitals=8, active_orbitals=12))(rhf)
    ci.run()

    _, _, l2_3, l3 = ci.make_average_cumulants(max_order=3)
    _, _, l2_2, none3 = ci.make_average_cumulants(max_order=2)
    assert l3 is not None
    assert none3 is None
    assert np.abs(l2_3 - l2_2).max() == 0.0

    with pytest.raises(ValueError):
        ci.make_average_cumulants(max_order=4)
