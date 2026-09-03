import numpy as np

from forte2 import GHF, RHF, System
from forte2.orbitals import mo_overlap, project_occupied_orbitals
from forte2.helpers.comparisons import approx
from forte2.base_classes import X2CParams


def _system(delta, basis_set="cc-pvdz", x2c=None):
    xyz = f"""
    O 0 0 {delta}
    H 0 0 1
    H 0 1 0.1
    """
    sys = System(
        xyz=xyz,
        basis_set=basis_set,
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=x2c,
    )
    return sys


def test_mo_overlap_rhf():
    sys_a = _system(0)
    hf_a = RHF(charge=0)(sys_a).run()

    sys_b = _system(0.01)
    hf_b = RHF(charge=0)(sys_b).run()

    ovlp = mo_overlap(hf_a.C[0][:, :5], sys_a, hf_b.C[0][:, :5], sys_b)
    # <psi_a | psi_b> = det(S_alpha) det(S_beta) == det(S)^2 for RHF
    assert np.linalg.det(ovlp) ** 2 == approx(0.9918900343683039)


def test_mo_overlap_ghf():
    x2c = X2CParams(x2c_type="so", x2c_model="1e")
    sys_a = _system(0, x2c=x2c)
    hf_a = GHF(charge=0)(sys_a).run()

    sys_b = _system(0.01, x2c=x2c)
    hf_b = GHF(charge=0)(sys_b).run()

    ovlp = mo_overlap(hf_a.C[0][:, :10], sys_a, hf_b.C[0][:, :10], sys_b)
    # phrase is arbitrary, |det(S)| is the meaningful quantity
    assert np.abs(np.linalg.det(ovlp)) == approx(0.9918884351285194)


def test_project_occupied_orbitals():
    sys_a = _system(0)
    hf_a = RHF(charge=0)(sys_a).run()

    sys_b = _system(0.001)
    hf_b = RHF(charge=0)(sys_b)
    mo_guess = project_occupied_orbitals(hf_a, hf_b)
    hf_b.C = mo_guess
    hf_b.run()


def test_basis_set_upcasting():
    sys_a = _system(0, "cc-pvdz")
    hf_a = RHF(charge=0)(sys_a).run()

    sys_b = _system(0, "cc-pvtz")
    hf_b = RHF(charge=0)(sys_b)
    mo_guess = project_occupied_orbitals(hf_a, hf_b)
    hf_b.C = mo_guess
    hf_b.run()


def test_project_previous_occupied_orbitals_to_new_geometry():
    old_system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    new_system = System(
        xyz="H 0 0 0\nH 0 0 1.8",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    old_rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(old_system)
    old_rhf.run()
    new_rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10)(new_system)

    projected = project_occupied_orbitals(old_rhf, new_rhf)

    assert projected is not None
    assert len(projected) == 1
    assert projected[0].shape == (new_system.nbf, new_system.nmo)

    # Check that the projected orbitals are orthonormal in the new basis
    np.testing.assert_allclose(
        mo_overlap(projected[0], new_system, projected[0]),
        np.eye(new_system.nmo),
        atol=1.0e-10,
    )
    # Check that the projected occupied orbital has a large overlap with the old one
    occupied_overlap = mo_overlap(
        projected[0][:, : new_rhf.na],
        new_system,
        old_rhf.C[0][:, : old_rhf.na],
        old_system,
    )
    assert abs(occupied_overlap[0, 0]) > 0.99
