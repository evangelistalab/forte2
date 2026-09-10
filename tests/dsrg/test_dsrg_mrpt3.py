from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.dsrg import DSRG_MRPT2, DSRG_MRPT3
from forte2.helpers.comparisons import approx

# Shared N2 setup for the forte1 cross-validation tests: r near equilibrium,
# CAS(6,6) with the 1s pair frozen out of the correlation treatment.
#   forte1: frozen_docc [2], restricted_docc [2], active [6]
#        -> forte2: core_orbitals=4, active_orbitals=6, frozen_core_orbitals=2
XYZ_NEAR_EQ = """
N 0.0 0.0 0.000
N 0.0 0.0 2.074
"""


def _n2_system(xyz):
    return System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
    )


def test_sf_mrpt3_n2_ss():
    """State-specific DSRG-MRPT3 with iterative reference relaxation.

    Cross-validated against forte1 (spin-adapted SA-MRDSRG, corr_level pt3,
    calc_type ss, relax_ref iterate, dsrg_s 1.0). The three columns of
    relax_energies are exactly forte1's Fixed Ref. / Relaxed Ref. / Eref
    columns, one row per macro-iteration, so this pins the whole trajectory
    rather than just the converged number.
    """
    system = _n2_system(XYZ_NEAR_EQ)
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    ci_solver = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=4,
        active_orbitals=6,
    )
    mc = MCOptimizer(ci_solver, e_tol=1e-10, g_tol=1e-8)(rhf)

    dsrg = DSRG_MRPT3(
        flow_param=1.0,
        frozen_core_orbitals=2,
        relax_reference="iterate",
        relax_maxiter=15,
        relax_tol=1e-8,
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0] == approx(
        [-109.264722718306, -109.265375992059, -109.089949331959]
    )
    assert dsrg.relax_energies[1] == approx(
        [-109.265158793517, -109.265158796772, -109.089234082017]
    )
    assert dsrg.relax_energies[2] == approx(
        [-109.265159057239, -109.265159057240, -109.089234905661]
    )
    assert dsrg.relax_energies[3] == approx(
        [-109.265159054478, -109.265159054478, -109.089234897005]
    )
    assert dsrg.E_relaxed_ref == approx(-109.265159054478)


def test_sf_mrpt3_n2_sa():
    """State-averaged DSRG-MRPT3 over 2 singlets + 1 triplet, equal weights.

    forte1 equivalent: avg_state [[0,1,2],[0,3,1]], avg_weight [[1,1],[1]]
    (both sides normalize by the weight sum, so this is 1/3 each),
    calc_type sa, relax_ref iterate.
    """
    system = _n2_system(XYZ_NEAR_EQ)
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    singlet = State(nel=14, multiplicity=1, ms=0.0)
    triplet = State(nel=14, multiplicity=3, ms=0.0)
    ci_solver = CISolver(
        states=[singlet, triplet],
        core_orbitals=4,
        active_orbitals=6,
        nroots=[2, 1],
        weights=[[1 / 3, 1 / 3], [1 / 3]],
    )
    mc = MCOptimizer(ci_solver, e_tol=1e-10, g_tol=1e-8)(rhf)

    dsrg = DSRG_MRPT3(
        flow_param=1.0,
        frozen_core_orbitals=2,
        relax_reference="iterate",
        relax_maxiter=15,
        relax_tol=1e-8,
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0] == approx(
        [-109.045056536377, -109.046138956129, -108.849726938953]
    )
    assert dsrg.relax_energies[1] == approx(
        [-109.046051350346, -109.046051354973, -108.848575256216]
    )
    assert dsrg.relax_energies[2] == approx(
        [-109.046051490701, -109.046051490701, -108.848577694612]
    )
    assert dsrg.relax_energies[3] == approx(
        [-109.046051490135, -109.046051490135, -108.848577688125]
    )

    # The individually relaxed roots, not just their weighted average.
    assert dsrg.relax_eigvals == approx(
        [-109.267356517779, -108.890875684372, -108.979922268254]
    )


def test_sf_mrpt3_energy_decomposition():
    """Pin the four energy contributions separately, not just their sum.

    A wrong prefactor in one commutator stage shifts the total but leaves the
    other three stages intact, so pinning the decomposition localizes a failure
    to a single term. Values are forte1's "DSRG-MRPT3 Energy Summary" for the
    fixed-reference (unrelaxed) pass of test_sf_mrpt3_n2_ss.
    """
    system = _n2_system(XYZ_NEAR_EQ)
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    ci_solver = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=4,
        active_orbitals=6,
    )
    mc = MCOptimizer(ci_solver, e_tol=1e-10, g_tol=1e-8)(rhf)

    dsrg = DSRG_MRPT3(flow_param=1.0, frozen_core_orbitals=2)(mc)
    dsrg.run()

    assert dsrg.ints["E"] == approx(-109.089949331959)
    assert dsrg.e_dsrg_mrpt2 == approx(-0.154917646142)
    assert dsrg.e_dsrg_mrpt3_1 == approx(-0.001125808471)
    assert dsrg.e_dsrg_mrpt3_2 == approx(-0.017468190096)
    assert dsrg.e_dsrg_mrpt3_3 == approx(-0.001261741638)
    assert dsrg.E_dsrg == approx(-109.264722718306)


def test_sf_mrpt3_pt2_term_matches_standalone_mrpt2():
    """DSRG-MRPT3's own second-order term must equal a standalone DSRG-MRPT2.

    Both evaluate the same contraction from the same once-renormalized
    integrals and first-order amplitudes, so this isolates the integrals,
    amplitudes and renormalization from the three third-order commutator
    stages: it passes as soon as those are right, long before the third-order
    kernels are.
    """
    system = _n2_system(XYZ_NEAR_EQ)
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    ci_solver = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=4,
        active_orbitals=6,
    )
    mc = MCOptimizer(ci_solver, e_tol=1e-10, g_tol=1e-8)(rhf)

    pt2 = DSRG_MRPT2(flow_param=1.0, frozen_core_orbitals=2)(mc)
    pt2.run()

    pt3 = DSRG_MRPT3(flow_param=1.0, frozen_core_orbitals=2)(mc)
    pt3.run()

    assert pt3.e_dsrg_mrpt2 == approx(pt2.E_dsrg - pt2.ints["E"])


def test_sf_mrpt3_matches_two_component():
    """A plain GHF reference makes the two-component formalism spin-free.

    So the spin-free DSRG-MRPT3 must reproduce RelDSRG_MRPT3 exactly. The
    reference values are the ones test_mrpt3_n2_nonrel pins for the
    two-component code on this same system, which makes this an in-repo
    oracle: it needs no external program and it is a far tighter check than
    the cross-code comparisons above.

    Spinor counts halve: RelCISolver(core_orbitals=8, active_orbitals=12) and
    frozen_core_orbitals=4 become 4 / 6 / 2 spatial orbitals here.
    """
    system = _n2_system("""
        N 0.0 0.0 0.0
        N 0.0 0.0 2.0
        """)
    rhf = RHF(charge=0)(system)

    ci_solver = CISolver(
        states=State(nel=14, multiplicity=1, ms=0.0),
        core_orbitals=4,
        active_orbitals=6,
    )
    mc = MCOptimizer(ci_solver)(rhf)

    dsrg = DSRG_MRPT3(
        frozen_core_orbitals=2,
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
