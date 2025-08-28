import pytest
import numpy as np

from forte2 import System, RHF, CI, State, ROHF
from forte2.ci import RelCI
from forte2.scf.scf_utils import convert_coeff_spatial_to_spinor
from forte2.helpers.comparisons import approx
from forte2.system.build_basis import BSE_AVAILABLE


def prepare_rhf_coeff_for_relci(rhf, system):
    rhf = rhf.run()
    C = convert_coeff_spatial_to_spinor(system, rhf.C)[0]
    nmo = C.shape[1]
    random_phase = np.diag(np.exp(1j * np.random.uniform(-np.pi, np.pi, size=nmo)))
    C = C @ random_phase
    rhf.C[0] = C
    system.two_component = True
    return rhf, system


def test_rel_gasci_rhf_1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf, system = prepare_rhf_coeff_for_relci(rhf, system)
    ci = RelCI(
        active_orbitals=[[0, 1], [2, 3]],
        states=State(nel=2, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[2]),
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.05643120731551)
    assert ci.E[0] == approx(-1.096071975854)


def test_rel_gasci_rhf_2():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT")

    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf, system = prepare_rhf_coeff_for_relci(rhf, system)
    system.two_component = True
    ci = RelCI(
        active_orbitals=[[0, 1], [2, 3]],
        states=State(nel=2, multiplicity=1, ms=0.0, gas_min=[1, 0], gas_max=[2, 1]),
        nroots=2,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.089283671174)
    assert ci.E[0] == approx(-1.089283671174)
    # TODO: Add assertion for second root when the one below is externally verified
    # assert ci.E[1] == approx(-0.671622137375)


@pytest.mark.skipif(not BSE_AVAILABLE, reason="Basis set exchange is not available")
def test_rel_gasci_rhf_3():
    xyz = """
    H  0.000000000000  0.000000000000 -0.375000000000
    H  0.000000000000  0.000000000000  0.375000000000
    """

    system = System(
        xyz=xyz,
        basis_set="sto-6g",
        auxiliary_basis_set="def2-universal-jkfit",
        auxiliary_basis_set_corr="def2-svp-rifit",
    )

    rhf = RHF(charge=0, econv=1e-12)(system)
    rhf, system = prepare_rhf_coeff_for_relci(rhf, system)
    system.two_component = True
    ci = RelCI(
        active_orbitals=[[0, 1], [2, 3]],
        states=State(nel=2, multiplicity=1, ms=0.0, gas_min=[0, 0], gas_max=[2, 2]),
    )(rhf)
    ci.run()

    assert rhf.E == approx(-1.124751148359)
    assert ci.E[0] == approx(-1.145766051194)


def test_rel_gasci_rhf_4():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-12, dconv=1e-8)(system)
    rhf, system = prepare_rhf_coeff_for_relci(rhf, system)
    system.two_component = True
    ci = RelCI(
        active_orbitals=(10, 4),
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[6, 0], gas_max=[10, 4]),
        econv=1e-12,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    assert ci.E[0] == approx(-76.029447292783)


def test_rel_gasci_rhf_5():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = RHF(charge=0, econv=1e-14, dconv=1e-8)(system)
    rhf, system = prepare_rhf_coeff_for_relci(rhf, system)
    system.two_component = True
    ci = RelCI(
        active_orbitals=(2, 12),
        states=State(nel=10, multiplicity=1, ms=0.0, gas_min=[0], gas_max=[1]),
        nroots=4,
        basis_per_root=10,
        ndets_per_guess=20,
        maxiter=200,
    )(rhf)
    ci.run()

    assert rhf.E == approx(-76.02146209548764)
    # triplet is lower in this case
    assert ci.E[3] == approx(-55.819535370117)


def test_rel_gasci_rohf_3():
    xyz = """
    O  0.000000000000  0.000000000000 -0.069592187400
    H  0.000000000000 -0.783151105291  0.552239257834
    H  0.000000000000  0.783151105291  0.552239257834
    """

    system = System(
        xyz=xyz, basis_set="cc-pvdz", auxiliary_basis_set="def2-universal-jkfit"
    )

    rhf = ROHF(charge=0, econv=1e-12, dconv=1e-8, ms=1.0)(system)
    rhf, system = prepare_rhf_coeff_for_relci(rhf, system)
    ci = RelCI(
        active_orbitals=(2, 12),
        states=State(nel=10, multiplicity=3, ms=1.0, gas_min=[0], gas_max=[1]),
    )(rhf)
    ci.run()

    assert rhf.E == approx(-75.78642207312076)
    assert ci.E[0] == approx(-56.130750582569)
