from forte2 import CISolver, MCOptimizer, RHF, State, System
from forte2.dsrg import DSRG_MRPT2, DSRG_MRPT3
from forte2.helpers.comparisons import approx

# All references are forte (version 1) SA-MRDSRG with corr_level pt3 and
# int_type df, on the same SA-CASSCF reference and auxiliary basis.

N2_XYZ = """
N 0.0 0.0 0.000
N 0.0 0.0 2.074
"""


def _n2_system(symmetry=False):
    return System(
        xyz=N2_XYZ,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        symmetry=symmetry,
    )


def test_sf_mrpt3_n2_ss():
    """State-specific CAS(6,6) in D2h with the 1s pair frozen and iterative relaxation.

    Each row of relax_energies is forte's Fixed Ref. / Relaxed Ref. / Eref for
    one macro-iteration, so this pins the whole trajectory.
    """
    system = _n2_system(symmetry=True)
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
    """Two singlets and one triplet with equal weights, iterative relaxation."""
    system = _n2_system()
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
    assert dsrg.relax_eigvals == approx(
        [-109.267356517779, -108.890875684372, -108.979922268254]
    )


def test_sf_mrpt3_hf_sa():
    """Four singlets, including the degenerate 1Pi pair, with no frozen core.

    Checks each energy term separately, and checks the second-order term against
    a standalone DSRG-MRPT2.
    """
    xyz = """
    H 0.000000000000 0.000000000000 -0.949624435830
    F 0.000000000000 0.000000000000  0.050375564170
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    ci_solver = CISolver(
        states=State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=1,
        active_orbitals=5,
        nroots=4,
    )
    mc = MCOptimizer(ci_solver, e_tol=1e-12, g_tol=1e-8)(rhf)

    # before DSRG-MRPT3, whose relaxation replaces the CI solver's integrals
    pt2 = DSRG_MRPT2(flow_param=0.5)(mc)
    pt2.run()

    dsrg = DSRG_MRPT3(flow_param=0.5, relax_reference="once")(mc)
    dsrg.run()

    assert dsrg.ints["E"] == approx(-99.710996991409)
    assert dsrg.e_dsrg_mrpt2 == approx(-0.202910166520726)
    assert dsrg.e_dsrg_mrpt3_1 == approx(0.000945335674803)
    assert dsrg.e_dsrg_mrpt3_2 == approx(0.008462068027113)
    assert dsrg.e_dsrg_mrpt3_3 == approx(0.000951007875896)
    assert dsrg.E_dsrg == approx(-99.903548746458)
    assert dsrg.E_relaxed_ref == approx(-99.903660549958)
    assert dsrg.relax_eigvals == approx(
        [-100.217849530679, -99.862515166532, -99.862515166531, -99.671762336091]
    )
    assert dsrg.e_dsrg_mrpt2 == approx(pt2.E_dsrg - pt2.ints["E"])


def test_sf_mrpt3_h2o_sa():
    """Three singlets and a triplet with unequal weights and no correlated core.

    The O 1s orbital is frozen and every other occupied orbital is active, so
    every block with a core index is empty.
    """
    xyz = """
    O  0.000000000000  0.000000000000 -0.068516219320
    H  0.000000000000 -0.790689573744  0.543701060715
    H  0.000000000000  0.790689573744  0.543701060715
    """
    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")
    rhf = RHF(charge=0, e_tol=1e-12)(system)

    singlet = State(nel=10, multiplicity=1, ms=0.0)
    triplet = State(nel=10, multiplicity=3, ms=0.0)
    ci_solver = CISolver(
        states=[singlet, triplet],
        core_orbitals=1,
        active_orbitals=6,
        nroots=[3, 1],
        weights=[[0.4, 0.2, 0.2], [0.2]],
    )
    mc = MCOptimizer(ci_solver, e_tol=1e-12, g_tol=1e-8)(rhf)

    dsrg = DSRG_MRPT3(
        flow_param=0.5,
        frozen_core_orbitals=1,
        relax_reference="iterate",
        relax_maxiter=15,
        relax_tol=1e-9,
    )(mc)
    dsrg.run()

    assert dsrg.relax_energies[0, :2] == approx([-76.048512058859, -76.049991408088])
    assert dsrg.relax_eigvals_history[0] == approx(
        [-76.237739180788, -75.941476259825, -75.864949047939, -75.968053371101]
    )
    assert dsrg.E_relaxed_ref == approx(-76.050218518092)
