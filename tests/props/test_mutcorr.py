from forte2 import CI, CISolver, RHF, State, System
from forte2.props import MutualCorrelationAnalysis
from forte2.helpers.comparisons import approx
from forte2.base_classes import DavidsonLiuParams
import numpy as np

def test_mutual_correlation_h2_singlet():
    """Test mutual correlation analysis on H2 molecule in STO-6G basis at dissociation."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 10.0
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)
    #mutual correlation wants to have the CI solver named "ci", not the executable run function. 
    ci = CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            active_orbitals=[0, 1],
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
    casci=CI(ci)(rhf)
    casci.run()

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
    ci = CISolver(
            State(system=system, multiplicity=3, ms=0.0),
            active_orbitals=[0, 1],
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
    casci=CI(ci)(rhf)
    casci.run()

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
    ci = CISolver(
            State(system=system, multiplicity=3, ms=1.0),
            active_orbitals=[0, 1],
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
    casci=CI(ci)(rhf)
    casci.run()

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
    ci = CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            active_orbitals=list(range(10)),
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
    casci=CI(ci)(rhf)
    casci.run()

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
    ci = CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            active_orbitals=list(range(6)),
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
    casci=CI(ci)(rhf)
    casci.run()

    mca = MutualCorrelationAnalysis(ci)
    assert mca.total_correlation == approx(0.815410515)
    assert mca.M2[2, 3] == approx(0.562132887)

    summary = mca.mutual_correlation_matrix_summary()
    assert float(summary.splitlines()[5].split()[-1]) == approx(0.562133)

# Regression test: ensure MutualCorrelationAnalysis accesses sub-solvers directly from the active-space solver.

class FakeSubSolver:
    def make_rdm(self, root, order, spin_type):
        assert root == 0
        assert spin_type == "sd"

        if order == 1:
            return np.zeros((2, 2)), np.zeros((2, 2))

        if order == 2:
            # The production code converts the aa and bb tensors with this
            # helper. The test patches that conversion below.
            zeros = np.zeros((2, 2, 2, 2))
            return zeros, zeros.copy(), zeros.copy()

        raise AssertionError(f"Unexpected RDM order: {order}")


class FakeSolver:
    def __init__(self):
        self.mo_space = type("MOSpace", (), {"active_indices": [0, 1]})()
        self.sub_solvers = [FakeSubSolver()]


class FakeDriver:
    def __init__(self):
        self.mo_space = type("MOSpace", (), {"active_indices": [0, 1]})()
        self.ci_solver = FakeSolver()


def test_mutual_correlation_uses_direct_sub_solvers(monkeypatch):
    """MutualCorrelationAnalysis should accept an active-space solver directly."""

    from forte2.props import mutual_correlation

    monkeypatch.setattr(
        mutual_correlation.cpp_helpers,
        "packed_tensor4_to_tensor4",
        lambda tensor: tensor,
    )

    analysis = MutualCorrelationAnalysis(FakeSolver())

    assert analysis.active_mo_indices == [0, 1]
    assert analysis.total_correlation == 0.0
    np.testing.assert_array_equal(analysis.M1, np.zeros(2))
    np.testing.assert_array_equal(analysis.M2, np.zeros((2, 2)))


def test_mutual_correlation_accepts_driver(monkeypatch):
    """MutualCorrelationAnalysis should accept CI/MC drivers that own a ci_solver."""

    from forte2.props import mutual_correlation

    monkeypatch.setattr(
        mutual_correlation.cpp_helpers,
        "packed_tensor4_to_tensor4",
        lambda tensor: tensor,
    )

    analysis = MutualCorrelationAnalysis(FakeDriver())

    assert analysis.active_mo_indices == [0, 1]
    assert analysis.total_correlation == 0.0
    np.testing.assert_array_equal(analysis.M1, np.zeros(2))
    np.testing.assert_array_equal(analysis.M2, np.zeros((2, 2)))
