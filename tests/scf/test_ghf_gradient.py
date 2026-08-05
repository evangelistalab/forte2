import numpy as np
import pytest

from forte2 import GHF, System, UHF
from tests.scf.gradient_test_utils import (
    _xyz,
    four_point_central_difference_gradient_component,
)


def _ghf(symbols, coordinates, charge=1, **kwargs):
    system = System(
        xyz=_xyz(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    return GHF(charge=charge, e_tol=1.0e-12, d_tol=1.0e-10, **kwargs)(system)


def _ghf_energy(symbols, coordinates, charge=1, **kwargs):
    return _ghf(symbols, coordinates, charge=charge, **kwargs).run().E


def test_ghf_gradient_h2_cation_finite_difference():
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    ghf = _ghf(
        symbols,
        coordinates,
        alpha_beta_mix=True,
        break_complex_symmetry=True,
    )
    gradient = ghf.gradient()

    for atom in range(2):
        numerical = four_point_central_difference_gradient_component(
            _ghf_energy,
            symbols,
            coordinates,
            atom,
            2,
            alpha_beta_mix=True,
            break_complex_symmetry=True,
        )
        assert gradient[atom, 2] == pytest.approx(numerical, abs=1.0e-8)

    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)
    assert np.linalg.norm(ghf.D[1]) > 1.0e-8


def test_ghf_gradient_collinear_limit_matches_uhf():
    symbols = ["O", "H", "H"]
    coordinates = np.array([[0.0, 0.0, -0.1], [0.0, -1.4, 0.9], [0.0, 1.4, 0.9]])
    system_ghf = System(
        xyz=_xyz(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )
    system_uhf = System(
        xyz=_xyz(symbols, coordinates),
        basis_set="cc-pVDZ",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
    )

    ghf_gradient = GHF(charge=1, ms_guess=0.5)(system_ghf).gradient()
    uhf_gradient = UHF(charge=1, ms=0.5)(system_uhf).gradient()

    assert ghf_gradient == pytest.approx(uhf_gradient, abs=1.0e-9)


def test_ghf_gradient_auto_runs_and_is_repeatable():
    ghf = _ghf(["H", "H"], np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]))

    first = ghf.gradient()
    second = ghf.gradient()

    assert ghf.executed
    assert first.shape == (2, 3)
    assert first == pytest.approx(second, abs=1.0e-12)


@pytest.mark.parametrize("x2c_type", ["sf", "so"])
def test_x2c_ghf_gradient_finite_difference(x2c_type, monkeypatch):
    def make_system(z):
        return System(
            xyz=f"O 0 0 0\nH 0 0 {z:.12f}\nH 0 1.4 0",
            basis_set="sto-3g",
            auxiliary_basis_set="def2-universal-JKFIT",
            unit="bohr",
            x2c_type=x2c_type,
            minao_basis_set=None,
        )

    options = dict(charge=1, ms_guess=0.5, e_tol=1.0e-10, d_tol=1.0e-6)
    ghf = GHF(**options)(make_system(1.5))
    ghf.run()

    def reject_full_hcore_derivative():
        raise AssertionError("The production gradient requested full X2C matrices.")

    monkeypatch.setattr(
        ghf.system.x2c_helper, "hcore_deriv", reject_full_hcore_derivative
    )
    gradient = ghf.gradient()
    step = 5.0e-4
    energies = [
        GHF(**options)(make_system(1.5 + scale * step)).run().E
        for scale in (-2.0, -1.0, 1.0, 2.0)
    ]
    numerical = (energies[0] - 8.0 * energies[1] + 8.0 * energies[2] - energies[3]) / (
        12.0 * step
    )

    assert gradient[1, 2] == pytest.approx(numerical, abs=3.0e-7)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)
