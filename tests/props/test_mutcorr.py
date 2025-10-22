import numpy as np

from forte2 import System, RHF, CI, State
from forte2.props import MutualCorrelationAnalysis
from forte2.helpers.comparisons import approx


def test_mutual_correlation_h2_singlet():
    """Test mutual correlation analysis on H2 molecule in STO-6G basis at dissociation."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 10.0
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(State(system=system, multiplicity=1, ms=0.0), active_orbitals=[0, 1])(rhf)
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

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(State(system=system, multiplicity=3, ms=0.0), active_orbitals=[0, 1])(rhf)
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

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(State(system=system, multiplicity=3, ms=1.0), active_orbitals=[0, 1])(rhf)
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

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0), active_orbitals=list(range(10))
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

    rhf = RHF(charge=0, econv=1e-12)(system)
    ci = CI(
        State(system=system, multiplicity=1, ms=0.0), active_orbitals=list(range(6))
    )(rhf)
    ci.run()

    mca = MutualCorrelationAnalysis(ci)
    assert mca.total_correlation == approx(0.815410515)
    assert mca.M2[2, 3] == approx(0.562132887)
