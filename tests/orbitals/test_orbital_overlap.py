import logging

import numpy as np
import pytest

from forte2 import GHF, RHF, System
from forte2.orbitals import mo_overlap, transfer_orbitals
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


def _assert_orthonormal(C, system):
    np.testing.assert_allclose(
        mo_overlap(C, system, C), np.eye(C.shape[1]), atol=1.0e-10
    )


def test_transfer_orbitals_follows_the_atoms():
    # A heavy-atom core only survives a large step if the orbitals move with
    # the nuclei rather than being projected from where they used to be.
    system = System(
        xyz="Br 0 0 0\nH 0 0 2.7",
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    hf = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system).run()
    C = hf.C[0]

    np.testing.assert_allclose(transfer_orbitals(C, system, system), C, atol=1.0e-10)

    displaced = system.with_geometry([[0.2, -0.1, 0.1], [0.0, 0.1, 2.9]])
    C_new = transfer_orbitals(C, system, displaced)
    assert C_new.shape == (displaced.nbf, displaced.nmo)
    _assert_orthonormal(C_new, displaced)

    converged = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(displaced).run()
    core_overlap = mo_overlap(converged.C[0][:, :1], displaced, C_new[:, :1])
    assert abs(core_overlap[0, 0]) > 0.999

    seeded = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(displaced)
    seeded.C = [C_new]
    seeded.run()
    assert seeded.E == approx(converged.E)


def test_transfer_orbitals_between_basis_sets():
    sys_dz = _system(0, "cc-pvdz")
    hf_dz = RHF(charge=0)(sys_dz).run()
    sys_tz = _system(0, "cc-pvtz")

    C_tz = transfer_orbitals(hf_dz.C[0], sys_dz, sys_tz)
    assert C_tz.shape == (sys_tz.nbf, sys_tz.nmo)
    _assert_orthonormal(C_tz, sys_tz)

    seeded = RHF(charge=0)(sys_tz)
    seeded.C = [C_tz]
    seeded.run()
    assert seeded.E == approx(RHF(charge=0)(sys_tz).run().E)

    # Back to the smaller basis: the trailing orbitals are dropped.
    C_dz = transfer_orbitals(C_tz, sys_tz, sys_dz)
    assert C_dz.shape == (sys_dz.nbf, sys_dz.nmo)
    _assert_orthonormal(C_dz, sys_dz)


def test_transfer_orbitals_warns_when_the_target_cannot_represent_them(caplog):
    source = System(
        xyz="H 0 0 0\nH 0 0 1.4",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    far_away = System(
        xyz="H 0 0 100\nH 0 0 101.4",
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    hf = RHF(charge=0)(source).run()

    with caplog.at_level(logging.CRITICAL):
        assert transfer_orbitals(hf.C[0], source, far_away) is None
    assert "Cannot transfer orbitals" in caplog.text


def test_transfer_orbitals_rejects_mismatched_coefficients():
    xyz = "H 0 0 0\nH 0 0 1.4"
    kwargs = dict(
        basis_set="sto-3g", auxiliary_basis_set="def2-universal-JKFIT", unit="bohr"
    )
    source = System(xyz=xyz, **kwargs)
    C = RHF(charge=0)(source).run().C[0]

    two_component = System(xyz=xyz, **kwargs)
    GHF(charge=0)(two_component)
    with pytest.raises(ValueError, match="two-component"):
        transfer_orbitals(C, source, two_component)
