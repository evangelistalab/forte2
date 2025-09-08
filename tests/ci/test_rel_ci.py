import numpy as np
import pytest

from forte2 import System, RHF, RelState, GHF, State
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.helpers.comparisons import approx
from forte2.ci import RelCI, CI
from forte2.props import get_1e_property


def test_rel_ci_h2():
    # equivalent to test_slater_rules::test_slater_rules_1_complex
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-12)(system)
    scf.run()
    C = convert_coeff_spatial_to_spinor(system, scf.C)[0]
    nmo = C.shape[1]
    random_phase = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, size=nmo)))
    C = C @ random_phase
    scf.C[0] = C
    system.two_component = True

    state = RelState(nel=2)
    ci = RelCI(states=state, active_orbitals=4, do_test_rdms=True)(scf)
    ci.run()
    assert ci.E[0] == approx(-1.096071975854)


def test_rel_ci_hf():
    # equivalent to test_slater_rules::test_slater_rules_2_complex
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    scf = RHF(charge=0, econv=1e-10)(system)
    scf.run()
    C = convert_coeff_spatial_to_spinor(system, scf.C)[0]
    nmo = C.shape[1]
    random_phase = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, size=nmo)))
    C = C @ random_phase
    scf.C[0] = C
    system.two_component = True

    ci = RelCI(
        states=RelState(nel=10),
        core_orbitals=2,
        active_orbitals=12,
        do_test_rdms=True,
    )(scf)
    ci.run()
    assert ci.E[0] == approx(-100.019788438077)


def test_rel_ci_hf_ghf():
    # cross-validated with the pyscf fci_dhf_slow solver using integrals from SpinorbitalIntegrals
    eref = -100.10065023157668
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c_type="so",
    )
    scf = GHF(charge=0)(system)
    ci = RelCI(
        states=RelState(nel=10),
        core_orbitals=2,
        active_orbitals=12,
        do_test_rdms=True,
        final_orbital="semicanonical",
    )(scf)
    ci.run()
    assert ci.E[0] == approx(eref)


def test_rel_ci_hf_transition_dipole_equivalence_to_rhf():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        # reorient=False,
        # x2c_type="so",
    )
    scf = GHF(charge=0)(system)
    ci = RelCI(
        states=RelState(nel=10),
        nroots=4,
        core_orbitals=2,
        active_orbitals=12,
        do_transition_dipole=True,
    )(scf)
    ci.run()
    assert np.abs(ci.tdm_per_solver[0][(0, 0)]) == pytest.approx(
        [0.0, 0.0, 0.756780349], abs=1e-6
    )
    assert np.abs(ci.tdm_per_solver[0][(1, 1)]) == pytest.approx(
        [0.0, 0.0, 0.721450697], abs=1e-6
    )


def test_rel_ci_hf_transition_dipole_ghf():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        x2c_type="so",
    )
    scf = GHF(charge=0)(system)
    ci = RelCI(
        states=RelState(nel=10),
        nroots=4,
        core_orbitals=2,
        active_orbitals=12,
        do_transition_dipole=True,
        ci_algorithm="exact",
    )(scf)
    ci.run()
    assert ci.E[0] == approx(-100.10065023157668)
    assert ci.E[1] == approx(-99.7875319545)

    assert np.abs(ci.fosc_per_solver[0][(0, 1)]) == pytest.approx(
        0.000971182707117118, abs=1e-6
    )
    assert np.abs(ci.fosc_per_solver[0][(0, 2)]) == pytest.approx(
        0.0019659547500647276, abs=1e-6
    )
    assert np.abs(ci.fosc_per_solver[0][(0, 3)]) == pytest.approx(
        0.0005518591365081285, abs=1e-6
    )
