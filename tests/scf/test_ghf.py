import numpy as np

from forte2 import GHF, System, UHF, X2CParams
from forte2.data.atom_data import LIGHT_SPEED
from forte2.helpers.comparisons import approx
from forte2.scf.scf_base import SCFBase


def test_equivalence_to_rhf():
    e_ghf = -128.488756188998
    s2_ghf = 0.0
    xyz = """
    Ne 0 0 0
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
    )

    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(e_ghf)
    assert scf.S2 == approx(s2_ghf)


def test_ghf_complex_perturbation():
    """
    This test checks that, for a system that's stable wrt to
    Sz and time-reversal symmetry breaking, the GHF algorithm will converge to
    the same solution as UHF, even the inital DM breaks Sz and time-reversal symmetries.
    """
    e_uhf = -75.649277913857
    s2_uhf = 0.756178428699

    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(
        xyz=xyz,
        basis_set="cc-pvqz",
        auxiliary_basis_set="cc-pvqz-jkfit",
    )

    scf = UHF(charge=1, ms=0.5)(system)
    scf.run()
    assert scf.E == approx(e_uhf)
    assert scf.S2 == approx(s2_uhf)

    scf = GHF(charge=1)(system)
    # this option breaks Sz in the initial guess
    scf.alpha_beta_mix = True
    # this option breaks time-reversal/complex conjugation symmetry in the initial guess
    scf.break_complex_symmetry = True
    scf.run()
    assert scf.E == approx(e_uhf)
    assert scf.S2 == approx(s2_uhf)


def test_j_adapted_ghf():
    # The two bases should yield the same result
    eref = -398.723752737950
    s2ref = 1.0001601212478493
    xyz = """
    S 0 0 0
    H 0 0 1.1"""
    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    scf = GHF(charge=1, j_adapt=False)(system)
    scf.run()
    assert scf.E == approx(eref)
    assert scf.S2 == approx(s2ref)

    system = System(
        xyz=xyz,
        basis_set="decon-cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
    )
    scf = GHF(charge=1, j_adapt=True)(system)
    scf.run()
    assert scf.E == approx(eref)
    assert scf.S2 == approx(s2ref)


def test_equivalence_to_high_spin_uhf():
    euhf = -37.686541301113
    s2uhf = 2.0063122057868483
    xyz = """
    C 0 0 0"""
    system = System(
        xyz=xyz,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )
    scf = GHF(charge=0, ms_guess=1.0)(system)
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_equivalence_to_uhf():
    # this is the same test as test_uhf.py::test_uhf_singlet
    euhf = -76.061466407194
    s2uhf = 0.0
    xyz = """
    O            0.000000000000     0.000000000000    -0.061664597388
    H            0.000000000000    -0.711620616369     0.489330954643
    H            0.000000000000     0.711620616369     0.489330954643
    """

    system = System(xyz=xyz, basis_set="cc-pVQZ", auxiliary_basis_set="cc-pVQZ-JKFIT")

    scf = GHF(charge=0)(system)
    scf.run()
    assert scf.E == approx(euhf)
    assert scf.S2 == approx(s2uhf)


def test_ghf_guess_independent_of_degenerate_eigenbasis(monkeypatch):
    """
    The Kramers pairs of the SO-X2C guess are degenerate, so the eigensolver may return
    any rotation within each pair. Under strong spin-orbit coupling, the open-shell GHF
    has several solutions that differ only in the direction of the spin, so the guess
    must not depend on that rotation.
    """
    xyz = """
    O 0 0 0
    H 0 0 1.5
    H 0 1.4 0"""

    def make_system():
        return System(
            xyz=xyz,
            basis_set="sto-3g",
            auxiliary_basis_set="def2-universal-JKFIT",
            unit="bohr",
            x2c=X2CParams(x2c_type="so", x2c_model="1e", light_speed=LIGHT_SPEED / 10),
            minao_basis_set=None,
        )

    options = dict(charge=1, ms_guess=0.5, e_tol=1.0e-10, d_tol=1.0e-6)
    reference = GHF(**options)(make_system())
    reference.run()

    rng = np.random.default_rng(0)
    eigh = SCFBase._eigh

    def eigh_with_rotated_pairs(self, F):
        eps, C, irreps = eigh(self, F)
        for i in range(0, len(eps), 2):
            Z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            C[:, i : i + 2] = C[:, i : i + 2] @ np.linalg.qr(Z)[0]
        return eps, C, irreps

    # GHF diagonalizes its Fock matrix with its own _eigh, so only the guess is rotated.
    monkeypatch.setattr(SCFBase, "_eigh", eigh_with_rotated_pairs)
    rotated = GHF(**options)(make_system())
    rotated.run()

    assert rotated.E == approx(reference.E)
    assert rotated.S2 == approx(reference.S2)
