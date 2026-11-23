from forte2 import CI, CISolver, DMRGSolver, RHF, State, System
from forte2.props import MutualCorrelationAnalysis
from forte2.helpers.comparisons import approx, approx_abs
from forte2.base_classes import DavidsonLiuParams, DMRGParams
from forte2.dmrg import BLOCK2_AVAILABLE
import numpy as np
import pytest


def test_mutual_correlation_h2_singlet():
    """Test mutual correlation analysis on H2 molecule in STO-6G basis at dissociation."""

    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 10.0
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, e_tol=1e-12)(system)

    ci = CI(
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            active_orbitals=[0, 1],
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
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
        CISolver(
            State(system=system, multiplicity=3, ms=0.0),
            active_orbitals=[0, 1],
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
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
        CISolver(
            State(system=system, multiplicity=3, ms=1.0),
            active_orbitals=[0, 1],
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
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
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            active_orbitals=list(range(10)),
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
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
        CISolver(
            State(system=system, multiplicity=1, ms=0.0),
            active_orbitals=list(range(6)),
            davidson_liu_params=DavidsonLiuParams(e_tol=1e-10, r_tol=1e-5),
        )
    )(rhf)

    ci.run()

    mca = MutualCorrelationAnalysis(ci)
    assert mca.total_correlation == approx(0.815410515)
    assert mca.M2[2, 3] == approx(0.562132887)

    summary = mca.mutual_correlation_matrix_summary()
    assert float(summary.splitlines()[5].split()[-1]) == approx(0.562133)


@pytest.mark.skipif(not BLOCK2_AVAILABLE, reason="block2 is not installed")
def test_mutual_correlation_h6_triplet_dmrg(tmp_path):
    """Test that mutual correlation analysis of a DMRG wavefunction matches CI for the
    high-spin H6 triplet, whose alpha and beta RDMs differ."""

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
    state = State(system=system, multiplicity=3, ms=1.0)

    ci_solver = CISolver(
        state,
        active_orbitals=list(range(6)),
        davidson_liu_params=DavidsonLiuParams(e_tol=1e-12, r_tol=1e-8),
    )
    ci = CI(ci_solver)(rhf)
    ci.run()

    # Six sites, so a bond dimension of a few hundred is exact, and perturbative noise
    # can't move a solve off an excited state it lands on
    dmrg_params = DMRGParams(
        bond_dims=[200] * 4 + [400] * 4,
        noises=[1e-4] * 4 + [1e-6] * 2 + [0.0],
        noise_type="DensityMatrix",
        davidson_tols=[1e-12] * 8,
        n_sweeps=12,
        scratch=str(tmp_path),
    )
    dmrg_solver = DMRGSolver(
        states=state, active_orbitals=list(range(6)), dmrg_params=dmrg_params
    )
    dmrg = CI(dmrg_solver)(rhf)
    dmrg.run()

    mca_ci = MutualCorrelationAnalysis(ci)
    mca_dmrg = MutualCorrelationAnalysis(dmrg)
    assert mca_dmrg.total_correlation == approx_abs(mca_ci.total_correlation, 1e-6)
    for measure in ("M1", "M2", "M3", "M4"):
        np.testing.assert_allclose(
            getattr(mca_dmrg, measure), getattr(mca_ci, measure), atol=1e-6
        )


def test_driver_type_error():
    import pytest

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

    with pytest.raises(TypeError, match="driver must be an ActiveSpaceDriver"):
        MutualCorrelationAnalysis(ci)
