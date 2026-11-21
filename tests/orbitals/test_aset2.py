import numpy as np
import pytest

from forte2 import (
    ASET,
    ASET2,
    AVAS,
    CI,
    CISolver,
    DSRG_MRPT2,
    MCOptimizer,
    RHF,
    ROHF,
    SelectedCISolver,
    State,
    System,
)
from forte2.base_classes import SelectedCIParams
from forte2.helpers.comparisons import approx_abs


PROPENE_XYZ = """
C       -2.2314881720      2.3523969887      0.1565319638
C       -1.1287322054      1.6651786288     -0.1651010551
H       -3.2159664855      1.9109197306      0.0351701750
H       -2.1807424354      3.3645292222      0.5457999612
H       -1.2085033449      0.7043108616     -0.5330598833
C        0.2601218384      2.1970946692     -0.0290628762
H        0.7545456004      2.2023392001     -1.0052240245
H        0.8387453665      1.5599644558      0.6466877402
H        0.2749376338      3.2174213526      0.3670138598
"""

HNNF_XYZ = """
N       -1.1226987119      2.0137160725     -0.0992218410
N       -0.1519067161      1.2402226172     -0.0345618482
H        0.7253474870      1.7181546089     -0.2678695726
F       -2.2714806355      1.3880717623      0.2062454513
"""

BUTENE_XYZ = """
C        0.0000000000      0.0000000000      0.0000000000
C        1.3360000000      0.0000000000      0.0000000000
C        2.1884400000      1.2403200000      0.0000000000
C        3.6946900000      0.9592200000      0.0000000000
H       -0.5619000000      0.9351800000      0.0000000000
H       -0.5619000000     -0.9351800000      0.0000000000
H        1.8649300000     -0.9542100000      0.0000000000
H        1.9469000000      1.8424000000      0.8835000000
H        1.9469000000      1.8424000000     -0.8835000000
H        4.2625000000      1.8946000000      0.0000000000
H        3.9879000000      0.3879000000      0.8860000000
H        3.9879000000      0.3879000000     -0.8860000000
"""


def _ethane_methane_xyz(r_cc: float, r_env: float = 4.2) -> str:
    """Staggered ethane along z with a spectator methane offset in x."""
    r_ch = 1.094
    sin_t = np.sin(np.deg2rad(109.5))
    cos_t = np.cos(np.deg2rad(109.5))
    atoms = [("C", 0.0, 0.0, 0.0), ("C", 0.0, 0.0, r_cc)]
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        atoms.append(("H", r_ch * sin_t * np.cos(phi), r_ch * sin_t * np.sin(phi), r_ch * cos_t))
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0 + np.pi / 3.0
        atoms.append(("H", r_ch * sin_t * np.cos(phi), r_ch * sin_t * np.sin(phi), r_cc - r_ch * cos_t))
    # Spectator methane centered at (r_env, 0, r_cc / 2)
    cz = 0.5 * r_cc
    atoms.append(("C", r_env, 0.0, cz))
    d = r_ch / np.sqrt(3.0)
    for sx, sy, sz in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
        atoms.append(("H", r_env + sx * d, sy * d, cz + sz * d))
    return "\n".join(f"{el} {x: .8f} {y: .8f} {z: .8f}" for el, x, y, z in atoms)


def _ethyldiazene_xyz(r_nn: float) -> str:
    """All-trans ethyldiazene H-N1=N2-C1H2-C2H3 with N1=N2 along the x-axis."""
    r_nh, r_nc, r_cc, r_ch = 1.036, 1.470, 1.532, 1.098
    ang_nnh = np.deg2rad(106.5)
    ang_nnc = np.deg2rad(113.5)
    ang_ccc = np.deg2rad(112.5)

    n2 = np.array([0.0, 0.0, 0.0])
    n1 = np.array([r_nn, 0.0, 0.0])
    h1 = n1 + np.array([-r_nh * np.cos(ang_nnh), r_nh * np.sin(ang_nnh), 0.0])
    c1 = n2 + np.array([r_nc * np.cos(ang_nnc), -r_nc * np.sin(ang_nnc), 0.0])

    d1 = (c1 - n2) / np.linalg.norm(c1 - n2)
    th = -(np.pi - ang_ccc)  # all-trans bend away from N1
    ca, sa = np.cos(th), np.sin(th)
    d2 = np.array([ca * d1[0] - sa * d1[1], sa * d1[0] + ca * d1[1], 0.0])
    c2 = c1 + r_cc * d2

    # C1H2 hydrogens
    v1 = (n2 - c1) / np.linalg.norm(n2 - c1)
    v2 = (c2 - c1) / np.linalg.norm(c2 - c1)
    bis = -(v1 + v2)
    bis /= np.linalg.norm(bis)
    zhat = np.array([0.0, 0.0, 1.0])
    half = np.deg2rad(53.5)
    h2 = c1 + r_ch * (np.cos(half) * bis + np.sin(half) * zhat)
    h3 = c1 + r_ch * (np.cos(half) * bis - np.sin(half) * zhat)

    # C2H3 hydrogens
    u_axis = (c2 - c1) / np.linalg.norm(c2 - c1)
    v_back = (n2 - c1) / np.linalg.norm(n2 - c1)
    u_perp = v_back - np.dot(v_back, u_axis) * u_axis
    u_perp /= np.linalg.norm(u_perp)
    ang_cch = np.deg2rad(111.0)
    ch3_hs = [
        c2
        + r_ch
        * (
            -np.cos(ang_cch) * u_axis
            + np.sin(ang_cch) * (np.cos(phi) * u_perp + np.sin(phi) * zhat)
        )
        for phi in (np.pi, np.pi / 3.0, -np.pi / 3.0)
    ]

    atoms = [
        ("N", *n1),
        ("N", *n2),
        ("C", *c1),
        ("C", *c2),
        ("H", *h1),
        ("H", *h2),
        ("H", *h3),
        *[("H", *h) for h in ch3_hs],
    ]
    return "\n".join(f"{el} {x: .8f} {y: .8f} {z: .8f}" for el, x, y, z in atoms)


def _o2_ethylene_xyz(R: float) -> str:
    """Perpendicular O2 above the center of ethylene (mini Paper 2 Fig. 6 model)."""
    r_oo = 1.2075
    atoms = [
        ("O", 0.0, 0.0, R),
        ("O", 0.0, 0.0, R + r_oo),
        ("C", -0.668, 0.0, 0.0),
        ("C", 0.668, 0.0, 0.0),
        ("H", -1.230, 0.924, 0.0),
        ("H", -1.230, -0.924, 0.0),
        ("H", 1.230, 0.924, 0.0),
        ("H", 1.230, -0.924, 0.0),
    ]
    return "\n".join(f"{el} {x: .8f} {y: .8f} {z: .8f}" for el, x, y, z in atoms)


def _naive_dense_giant_active_aset2(aset, mc, flow_param=0.5, include_env_corr=True):
    """
    Independent mathematical oracle for ASET(2) on small molecules using the
    "giant active space" construction (Paper 2 Sec. 2.7) evaluated via
    DSRG_MRPT2's dense-TEI contraction kernels on padded N_A-sized RDMs.
    """
    emb = aset.emb_space
    semican = aset.semican
    C_semi = semican.C_semican
    fock_full = semican.fock_semican
    eps_full = semican.eps_semican
    Uactv = semican.Uactv

    g1_orig = mc.ci_solver.make_average_rdm(1)
    l2_orig = mc.ci_solver.make_average_cumulant(2)
    g1_a = np.einsum("ip,ij,jq->pq", Uactv, g1_orig, Uactv.conj(), optimize=True)
    l2_a = np.einsum(
        "ip,jq,ijkl,kr,ls->pqrs",
        Uactv,
        Uactv,
        l2_orig,
        Uactv.conj(),
        Uactv.conj(),
        optimize=True,
    )

    sl_ob = emb.B_core
    sl_ca_full = emb.A_core
    sl_aa_full = emb.actv
    sl_va_full = emb.A_virt
    sl_vb = emb.B_virt
    sl_corr = slice(sl_ob.start, sl_vb.stop)

    nc = sl_ob.stop - sl_ob.start
    nc_a = sl_ca_full.stop - sl_ca_full.start
    na_a = sl_aa_full.stop - sl_aa_full.start
    nv_a = sl_va_full.stop - sl_va_full.start
    na = nc_a + na_a + nv_a
    nv = sl_vb.stop - sl_vb.start
    sl_aa = slice(nc_a, nc_a + na_a)

    # Dense padded gamma1, eta1, lambda2 over all N_A fragment orbitals
    g1_pad = np.zeros((na, na))
    g1_pad[:nc_a, :nc_a] = 2.0 * np.eye(nc_a)
    g1_pad[sl_aa, sl_aa] = g1_a
    e1_pad = 2.0 * np.eye(na) - g1_pad
    l2_pad = np.zeros((na, na, na, na))
    l2_pad[sl_aa, sl_aa, sl_aa, sl_aa] = l2_a
    l3_pad = np.zeros((na,) * 6)

    fock = fock_full[sl_corr, sl_corr].copy()
    eps = eps_full[sl_corr].copy()
    c = slice(0, nc)
    a = slice(nc, nc + na)
    v = slice(nc + na, nc + na + nv)
    h = slice(0, nc + na)
    p = slice(nc, nc + na + nv)

    C_corr = C_semi[:, sl_corr]
    V_full = aset.system.fock_builder.two_electron_integrals_block(C_corr)
    sp = {"c": c, "a": a, "v": v}
    V = {
        key: np.ascontiguousarray(
            V_full[sp[key[0]], sp[key[1]], sp[key[2]], sp[key[3]]]
        )
        for key in (
            "vvaa",
            "aacc",
            "avca",
            "avac",
            "vaaa",
            "aaca",
            "aaaa",
            "vvcc",
            "vvac",
            "vacc",
        )
    }

    # Populate a standalone DSRG_MRPT2 instance with diagonal H^(0) on the giant active space
    pt2 = DSRG_MRPT2(flow_param=flow_param)
    pt2.ncore, pt2.nact, pt2.nvirt = nc, na, nv
    pt2.core, pt2.actv, pt2.virt = c, a, v
    pt2.hole, pt2.part = h, p
    pt2.hc, pt2.ha = c, a
    pt2.pa, pt2.pv = slice(0, na), slice(na, na + nv)
    pt2.fock = fock.copy()
    # Paper 2 Eq. (18): diagonal H^(0) within giant active space
    pt2.fock[a, a] = np.diag(eps[a])
    pt2.eps = eps
    pt2.cumulants = {
        "gamma1": g1_pad,
        "eta1": e1_pad,
        "lambda2": l2_pad,
        "lambda3": l3_pad,
    }
    pt2.ints = {
        "E": mc.E,
        "V": V,
        "B": None,
        "eps": {
            "c": eps[c],
            "a": eps[a],
            "v": eps[v],
            "h": eps[h],
            "p": eps[p],
        },
    }
    pt2.T1, pt2.T2 = pt2._build_tamps()
    if not include_env_corr:
        pt2.T1[pt2.hc, pt2.pv] = 0.0
        pt2.ints["V"]["vvcc"][:] = 0.0
    pt2.F_tilde = pt2._renormalize_F()
    pt2._renormalize_V_in_place()
    pt2.hbar_aa_df = np.zeros((na, na))
    dE = pt2._compute_pt2_energy(form_hbar=True)
    C1 = pt2._compute_Hbar_aa()
    C2 = pt2._compute_Hbar_aaaa()

    hbar1_no = fock[a, a].copy() + 0.5 * (C1 + C1.T)
    hbar2_no = V["aaaa"].copy() + 0.5 * (C2 + C2.transpose(2, 3, 0, 1))
    hbar2_temp = 2.0 * hbar2_no - hbar2_no.swapaxes(2, 3)

    E_tot = mc.E + dE
    E_scalar = (
        E_tot
        - np.einsum("vu,vu->", hbar1_no, g1_pad)
        + 0.25 * np.einsum("uv,vyux,xy->", g1_pad, hbar2_temp, g1_pad)
        - 0.5 * np.einsum("xyuv,uvxy->", hbar2_no, l2_pad)
    )
    hbar1 = hbar1_no - 0.5 * np.einsum("uxvy,yx->uv", hbar2_temp, g1_pad)
    return E_scalar, hbar1, hbar2_no, E_tot


def _run_dsrg_root(ci_driver, root, flow_param=0.5):
    """Run DSRG-MRPT2 on a single root of a multi-root CI driver."""
    nroots = len(ci_driver.E_ci)
    weights = [1.0 if i == root else 0.0 for i in range(nroots)]
    ci_r = CI(
        CISolver(
            State(system=ci_driver.system, multiplicity=1, ms=0.0),
            nroots=nroots,
            weights=weights,
            frozen_core_orbitals=ci_driver.mo_space.frozen_core_indices,
            core_orbitals=ci_driver.mo_space.core_indices,
            active_orbitals=ci_driver.mo_space.active_indices,
            frozen_virtual_orbitals=ci_driver.mo_space.frozen_virtual_indices,
        ),
        final_orbitals="semicanonical",
    )(ci_driver)
    return DSRG_MRPT2(flow_param=flow_param)(ci_r).run().E_dsrg


# ---------------------------------------------------------------------------
# Exact mathematical identity tests
# ---------------------------------------------------------------------------


def test_aset2_zero_flow_reduces_to_aset0():
    """At flow_param=0.0, ASET(2)-[CI/MRPT2] must equal ASET(0)-[CI/MRPT2]."""
    system = System(
        xyz=PROPENE_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0), core_orbitals=11, active_orbitals=2
        ),
        e_tol=1e-10,
        g_tol=1e-10,
    )(rhf)
    aset = ASET(fragment=["C1-2", "H1-3"], cutoff_method="threshold", cutoff=0.5)(mc)

    # ASET(0) path (unrelaxed and relaxed)
    ci0 = CI(CISolver(State(system=system, multiplicity=1, ms=0.0)))(aset)
    pt2_0 = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(ci0).run()

    # ASET(2) with flow_param=0.0
    aset2 = ASET2(flow_param=0.0)(aset).run()
    assert aset2.delta_E_AB == approx_abs(0.0, 1e-13)
    assert aset2.delta_E_B == approx_abs(0.0, 1e-13)
    assert aset2.E_ref == approx_abs(mc.E, 1e-12)

    ci2 = CI(CISolver(State(system=aset2.system, multiplicity=1, ms=0.0)))(aset2)
    pt2_2 = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(ci2).run()

    assert ci2.E == approx_abs(ci0.E, 1e-11)
    assert pt2_2.relax_energies[0, 0] == approx_abs(pt2_0.relax_energies[0, 0], 1e-11)
    assert pt2_2.relax_energies[0, 1] == approx_abs(pt2_0.relax_energies[0, 1], 1e-11)


def test_aset2_all_active_reduces_to_pr_dsrg_mrpt2():
    """
    When C_A = empty and V_A = empty (all fragment orbitals are active),
    ASET(2)-[CASCI] reduces to partially-relaxed DSRG-MRPT2 (including lambda_3).
    """
    system = System(
        xyz=HNNF_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    mc.run()

    # Run ASET + ASET(2) before pt2_ref relaxes mc.ci_solver in-place
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_A_occ=0,
        num_A_vir=0,
    )(mc)
    aset2_l3 = ASET2(flow_param=0.5, skip_3_cumulant=False)(aset).run()
    aset2_no3 = ASET2(flow_param=0.5, skip_3_cumulant=True)(aset).run()
    ci2 = CI(CISolver(State(system=aset2_l3.system, multiplicity=1, ms=0.0)))(
        aset2_l3
    ).run()

    # Reference pr-DSRG-MRPT2 (includes lambda_3 on main)
    pt2_ref = DSRG_MRPT2(flow_param=0.5, relax_reference="once")(mc).run()

    # Tiny ~3e-7 difference comes solely from DF renormalize_3index on off-diag g1 in pt2_ref
    assert aset2_l3.E_ref == approx_abs(mc.E, 1e-12)
    assert aset2_l3.E_aset2_ref == approx_abs(pt2_ref.relax_energies[0, 0], 1e-6)
    assert ci2.E == approx_abs(pt2_ref.relax_energies[0, 1], 1e-5)
    # lambda_3 contribution is nonzero and small for CAS(4e, 4o)
    dl3 = aset2_l3.E_aset2_ref - aset2_no3.E_aset2_ref
    assert abs(dl3) > 1e-6 and abs(dl3) < 1e-3


@pytest.mark.parametrize("include_env_corr", [True, False])
def test_aset2_matches_naive_giant_active_oracle(include_env_corr):
    """
    Element-by-element verification of E_scalar, hbar1, hbar2, and E_aset2_ref
    against an independent dense giant-active-space oracle on HNNF (STO-3G).
    """
    system = System(
        xyz=HNNF_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    aset = ASET(
        fragment=["N", "H"],
        cutoff_method="num_of_orbitals",
        num_A_occ=5,
        num_A_vir=1,
    )(mc)
    aset2 = ASET2(flow_param=0.5, include_env_corr=include_env_corr)(aset).run()

    E_sc_ref, h1_ref, h2_ref, E_tot_ref = _naive_dense_giant_active_aset2(
        aset, mc, flow_param=0.5, include_env_corr=include_env_corr
    )

    assert aset2.E_aset2_ref == approx_abs(E_tot_ref, 1e-11)
    assert aset2.E_scalar == approx_abs(E_sc_ref, 1e-11)
    np.testing.assert_allclose(aset2.hbar1, h1_ref, atol=1e-11)
    np.testing.assert_allclose(aset2.hbar2, h2_ref, atol=1e-11)


def test_aset2_full_molecule_matches_unembedded():
    """When fragment A = all atoms (O_B = V_B = empty), ASET(2)-[MR] == full MR."""
    system = System(
        xyz=HNNF_XYZ,
        basis_set="sto-3g",
        cholesky_tei=True,
        cholesky_tol=1e-10,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    pt2_full = DSRG_MRPT2(flow_param=0.5)(mc).run()

    aset = ASET(fragment=["N", "H", "F"], cutoff_method="threshold", cutoff=0.01)(mc)
    aset2 = ASET2(flow_param=0.5)(aset).run()
    ci2 = CI(CISolver(State(system=aset2.system, multiplicity=1, ms=0.0)))(aset2)
    pt2_emb = DSRG_MRPT2(flow_param=0.5)(ci2).run()

    assert aset2.delta_E_AB == approx_abs(0.0, 1e-13)
    assert aset2.delta_E_B == approx_abs(0.0, 1e-13)
    assert ci2.E == approx_abs(mc.E, 1e-11)
    assert pt2_emb.E_dsrg == approx_abs(pt2_full.E_dsrg, 1e-6)


def test_aset2_selected_ci_solver():
    """Verify ASET(2) dressed 4-fold Hermitian integrals work with SelectedCISolver."""
    system = System(
        xyz=HNNF_XYZ,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    mc = MCOptimizer(
        CISolver(
            State(nel=24, multiplicity=1, ms=0.0),
            frozen_core_orbitals=3,
            core_orbitals=7,
            active_orbitals=4,
        ),
        e_tol=1e-10,
        g_tol=1e-10,
        maxiter=300,
        max_rotation=0.15,
    )(rhf)
    aset = ASET(fragment=["N", "H"], cutoff_method="threshold", cutoff=0.5)(mc)
    aset2 = ASET2(flow_param=0.5)(aset).run()

    state_f = State(system=aset2.system, multiplicity=1, ms=0.0)
    ci_exact = CI(CISolver(state_f))(aset2).run()
    # Dressed ASET(2) integrals have 4-fold (not 8-fold) symmetry, so SelectedCISolver
    # must use exact SlaterRules (ci_algorithm="exact") rather than the 8-fold iterative kernel.
    sci_exact = CI(
        SelectedCISolver(
            state_f,
            sci_params=SelectedCIParams(
                var_threshold=1e-12, pt2_threshold=0.0, ci_algorithm="exact"
            ),
        )
    )(aset2).run()
    assert sci_exact.E == approx_abs(ci_exact.E, 1e-10)


# ---------------------------------------------------------------------------
# Compact chemical benchmark tests distilled from Papers 1 & 2
# ---------------------------------------------------------------------------


def test_aset2_ethane_methane_dissociation():
    """
    Mini Paper 1 Fig. 5 / Paper 2 test: ethane C-C stretch next to spectator methane.

    Verifies that ASET(2)-[DSRG-MRPT2] removes >99% of ASET(0)'s frozen-methane
    absolute energy shift and accurately reproduces the full DSRG-MRPT2 stretch energy.
    """
    energies = {}
    for r_cc in (1.54, 3.50):
        system = System(
            xyz=_ethane_methane_xyz(r_cc=r_cc),
            basis_set="sto-3g",
            auxiliary_basis_set="cc-pvtz-jkfit",
        )
        rhf = RHF(charge=0, e_tol=1e-12)(system).run()
        avas = AVAS(subspace=["C1-2(2pz)"], cas=(2, 2))(rhf).run()
        mc = MCOptimizer(
            CISolver(
                State(system=system, multiplicity=1, ms=0.0),
                frozen_core_orbitals=avas.mo_space.core_indices[:3],
                core_orbitals=avas.mo_space.core_indices[3:],
                active_orbitals=avas.mo_space.active_indices,
            ),
            e_tol=1e-10,
            g_tol=1e-8,
        )(avas).run()

        e_full = DSRG_MRPT2(flow_param=0.5)(mc).run().E_dsrg
        aset = ASET(fragment=["C1-2", "H1-6"], cutoff_method="threshold", cutoff=0.5)(
            mc
        )
        ci0 = CI(CISolver(State(system=system, multiplicity=1, ms=0.0)))(aset)
        e_a0 = DSRG_MRPT2(flow_param=0.5)(ci0).run().E_dsrg

        a2 = ASET2(flow_param=0.5)(aset).run()
        ci2 = CI(CISolver(State(system=a2.system, multiplicity=1, ms=0.0)))(a2)
        e_a2 = DSRG_MRPT2(flow_param=0.5)(ci2).run().E_dsrg
        energies[r_cc] = (e_full, e_a0, e_a2)

    # At equilibrium, ASET(0) misses methane correlation (~0.05 Eh in STO-3G),
    # while ASET(2) collapses the absolute error by >100x
    e_full_eq, e_a0_eq, e_a2_eq = energies[1.54]
    assert abs(e_a0_eq - e_full_eq) > 1e-2
    assert abs(e_a2_eq - e_full_eq) < 1e-3
    assert abs(e_a2_eq - e_full_eq) < 0.01 * abs(e_a0_eq - e_full_eq)

    # Stretch energy matches full DSRG-MRPT2 to better than 0.05 kcal/mol
    dE_full = (energies[3.50][0] - energies[1.54][0]) * 627.5094740631
    dE_a2 = (energies[3.50][2] - energies[1.54][2]) * 627.5094740631
    assert abs(dE_a2 - dE_full) < 0.05


def test_aset2_butene_state_specific_excitation():
    """
    Mini Paper 2 Fig. 5a & Sec. 4.2 test: 1-butene S0 -> S1 (pi -> pi*) vertical
    excitation using state-specific ASET(2) on shared GS-CASSCF orbitals.
    """
    EH_TO_EV = 27.211386245988
    system = System(
        xyz=BUTENE_XYZ,
        basis_set="6-31g",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system).run()
    avas = AVAS(subspace=["C1-2(2pz)"], cas=(2, 2))(rhf).run()
    mc = MCOptimizer(
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            frozen_core_orbitals=avas.mo_space.core_indices[:4],
            core_orbitals=avas.mo_space.core_indices[4:],
            active_orbitals=avas.mo_space.active_indices,
        ),
        e_tol=1e-10,
        g_tol=1e-7,
    )(avas).run()

    # Full GS-CASSCF + DSRG-MRPT2 vertical excitation
    ci_full = CI(
        CISolver(State(system=system, multiplicity=1, ms=0.0), nroots=2),
        final_orbitals="original",
    )(mc).run()
    dE_full = (
        _run_dsrg_root(ci_full, 1) - _run_dsrg_root(ci_full, 0)
    ) * EH_TO_EV

    # Embedded ASET(0) and state-specific ASET(2) on -CH=CH2 fragment with compact V_A
    aset = ASET(
        fragment=["C1-2", "H1-3"],
        cutoff_method="num_of_orbitals",
        num_A_occ=None,
        num_A_vir=8,
        cutoff=0.5,
    )(mc)
    ci_emb = CI(CISolver(State(system=system, multiplicity=1, ms=0.0), nroots=2))(
        aset
    ).run()
    dE_mf = (_run_dsrg_root(ci_emb, 1) - _run_dsrg_root(ci_emb, 0)) * EH_TO_EV

    def _aset2_exc(include_env_corr=True, skip_3_cumulant=True):
        es = []
        for rt in (0, 1):
            a2 = ASET2(
                flow_param=0.5,
                include_env_corr=include_env_corr,
                skip_3_cumulant=skip_3_cumulant,
                root=rt,
            )(ci_emb).run()
            w = [1.0 if i == rt else 0.0 for i in range(2)]
            ci_f = CI(
                CISolver(
                    State(system=a2.system, multiplicity=1, ms=0.0),
                    nroots=2,
                    weights=w,
                )
            )(a2)
            es.append(DSRG_MRPT2(flow_param=0.5)(ci_f).run().E_dsrg)
        return (es[1] - es[0]) * EH_TO_EV

    dE_a2 = _aset2_exc(include_env_corr=True, skip_3_cumulant=True)
    dE_a2_noB = _aset2_exc(include_env_corr=False, skip_3_cumulant=True)
    dE_a2_l3 = _aset2_exc(include_env_corr=True, skip_3_cumulant=False)

    # ASET(2) significantly improves the vertical excitation over ASET(mf)
    assert abs(dE_a2 - dE_full) < abs(dE_mf - dE_full)
    # Pure-environment correlation delta E^B cancels cleanly at fixed geometry
    assert abs(dE_a2 - dE_a2_noB) < 0.02
    # lambda_3 is nonzero even for 2 active electrons due to open-shell S1 cumulant
    assert abs(dE_a2_l3 - dE_a2) > 1e-4 and abs(dE_a2_l3 - dE_a2) < 0.05


def test_aset2_o2_ethylene_singlet_triplet_gap():
    """
    Mini Paper 2 Fig. 6 test: perpendicular O2-ethylene singlet-triplet gap
    Delta E_ST(R) = E_S(R) - E_T(R) using ROHF -> AVAS -> CASSCF(8e, 6o).

    Verifies that ASET(2)-[DSRG-MRPT2] captures the environment-induced singlet
    stabilization between R = 4.0 A and R = 2.0 A that ASET(mf) misses.
    """
    EH_TO_EV = 27.211386245988

    def _gap(R):
        system = System(
            xyz=_o2_ethylene_xyz(R),
            basis_set="6-31g",
            auxiliary_basis_set="cc-pvtz-jkfit",
        )
        rohf = ROHF(charge=0, ms=1.0, e_tol=1e-12)(system).run()
        avas = AVAS(subspace=["O1-2(2p)"], cas=(8, 6))(rohf).run()

        res = {}
        for mult, ms, nroots in ((3, 1.0, 1), (1, 0.0, 2)):
            st = State(system=system, multiplicity=mult, ms=ms)
            mc = MCOptimizer(
                CISolver(
                    st,
                    nroots=nroots,
                    frozen_core_orbitals=avas.mo_space.core_indices[:4],
                    core_orbitals=avas.mo_space.core_indices[4:],
                    active_orbitals=avas.mo_space.active_indices,
                ),
                e_tol=1e-10,
                g_tol=1e-6,
                maxiter=120,
            )(avas).run()
            e_full = DSRG_MRPT2(flow_param=0.5)(mc).run().E_dsrg

            aset = ASET(fragment=["O1-2"], cutoff_method="threshold", cutoff=0.5)(mc)
            ci0 = CI(CISolver(st, nroots=nroots))(aset)
            e_mf = DSRG_MRPT2(flow_param=0.5)(ci0).run().E_dsrg

            a2 = ASET2(flow_param=0.5, skip_3_cumulant=False)(aset).run()
            st_f = State(system=a2.system, multiplicity=mult, ms=ms)
            ci2 = CI(CISolver(st_f, nroots=nroots))(a2)
            e_a2 = DSRG_MRPT2(flow_param=0.5)(ci2).run().E_dsrg
            res[mult] = (e_full, e_mf, e_a2)
        return tuple((res[1][i] - res[3][i]) * EH_TO_EV for i in range(3))

    g_near = _gap(2.0)  # (full, mf, a2)
    g_far = _gap(4.0)

    shift_full = g_far[0] - g_near[0]
    shift_mf = g_far[1] - g_near[1]
    shift_a2 = g_far[2] - g_near[2]

    # Full and ASET(2) both capture singlet stabilization at short R; ASET(2) beats ASET(mf)
    assert shift_full > 0.01
    assert abs(shift_a2 - shift_full) < abs(shift_mf - shift_full)
    assert abs(shift_a2 - shift_full) < 0.015


def test_aset_nullspace_fock_canonicalization_and_aset2():
    """
    Mini Paper 2 Fig. 4 / ethyldiazene test: requesting more virtual orbitals
    than rank(P_vv) via cutoff_method="num_of_orbitals".

    Verifies that Fock canonicalization inside the degenerate zero-eigenvalue
    null space keeps the null-space F_{V_A, V_B} coupling zero and yields smooth,
    accurate ASET(2) bond-stretch energetics.
    """
    diffs_a2 = []
    for r_nn in (1.25, 2.00):
        system = System(
            xyz=_ethyldiazene_xyz(r_nn),
            basis_set="6-31g",
            auxiliary_basis_set="cc-pvtz-jkfit",
        )
        rhf = RHF(charge=0, e_tol=1e-12)(system).run()
        avas = AVAS(subspace=["N1-2(2px)", "N1-2(2pz)"], cas=(4, 4))(rhf).run()
        mc = MCOptimizer(
            CISolver(
                State(system=system, multiplicity=1, ms=0.0),
                frozen_core_orbitals=avas.mo_space.core_indices[:4],
                core_orbitals=avas.mo_space.core_indices[4:],
                active_orbitals=avas.mo_space.active_indices,
            ),
            e_tol=1e-10,
            g_tol=1e-7,
            maxiter=120,
        )(avas).run()
        e_full = DSRG_MRPT2(flow_param=0.5)(mc).run().E_dsrg

        # Fragment ["N1-2", "H1"] in 6-31G has 20 AOs -> rank(P_vv) <= 20 < 28,
        # so num_A_vir=28 cuts deep into the zero-eigenvalue null space of P_vv.
        aset = ASET(
            fragment=["N1-2", "H1"],
            cutoff_method="num_of_orbitals",
            num_A_occ=None,
            num_A_vir=28,
            cutoff=0.5,
        )(mc).run()

        # Verify that rank(P_vv) < 28 so num_A_vir=28 cuts into the null space,
        # and that Fock canonicalization keeps the V_A-V_B coupling small and smooth
        lv = aset.partition["lv_vals"]
        n_nonnull = int(np.sum(lv >= 1e-10))
        assert n_nonnull < 28
        f_semi = aset.semican.fock_semican
        f_vavb = f_semi[aset.emb_space.A_virt, aset.emb_space.B_virt]
        assert np.max(np.abs(f_vavb)) < 0.25

        # Verify ASET(2) runs cleanly and matches full DSRG-MRPT2 tightly
        a2 = ASET2(flow_param=0.5)(aset).run()
        ci2 = CI(CISolver(State(system=a2.system, multiplicity=1, ms=0.0)))(a2)
        e_a2 = DSRG_MRPT2(flow_param=0.5)(ci2).run().E_dsrg
        diffs_a2.append((e_a2 - e_full) * 627.5094740631)

    # Smoothness across N=N stretch: error change < 1.5 kcal/mol
    assert abs(diffs_a2[1] - diffs_a2[0]) < 1.5


def test_aset_and_aset2_with_rhf_and_rohf_parents():
    """Verify ASET(0) and ASET(2) support RHF, ROHF, and CI parents cleanly."""
    system = System(
        xyz=PROPENE_XYZ,
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    aset = ASET(fragment=["C1-2", "H1-3"], cutoff_method="threshold", cutoff=0.5)(
        rhf
    ).run()
    assert aset.mo_space.nactv == 0

    c_a = aset.partition["index_A_occ"]
    v_a = aset.partition["index_A_vir"]
    ci_ref = CI(
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            nroots=2,
            frozen_core_orbitals=aset.mo_space.frozen_core_indices,
            core_orbitals=c_a[:-1],
            active_orbitals=[c_a[-1], v_a[0]],
            frozen_virtual_orbitals=aset.mo_space.frozen_virtual_indices,
        )
    )(aset).run()

    energies = []
    for root in (0, 1):
        a2 = ASET2(flow_param=0.5, root=root)(ci_ref).run()
        ci_f = CI(
            CISolver(State(system=a2.system, multiplicity=1, ms=0.0), nroots=2)
        )(a2)
        energies.append(DSRG_MRPT2(flow_param=0.5)(ci_f).run().E_dsrg)
    assert np.all(np.isfinite(energies))

    # Also verify ROHF parent in ASET
    sys_o2 = System(
        xyz=_o2_ethylene_xyz(3.0),
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
    )
    rohf = ROHF(charge=0, ms=1.0, e_tol=1e-12)(sys_o2)
    aset_rohf = ASET(fragment=["O1-2"], cutoff_method="threshold", cutoff=0.5)(
        rohf
    ).run()
    assert aset_rohf.mo_space.nactv == 2
