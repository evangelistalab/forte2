import numpy as np
import pytest

from forte2 import System
from forte2.scf import RHF, UHF
from tests.gradient_test_utils import (
    four_point_central_difference_gradient_component,
    make_test_system,
    six_point_central_difference_gradient_component,
)


def _uhf(symbols, coordinates, charge, ms):
    uhf = UHF(charge=charge, ms=ms, e_tol=1.0e-12, d_tol=1.0e-8, maxiter=100)(
        make_test_system(symbols, coordinates)
    )
    uhf.run()
    return uhf


def _uhf_energy(symbols, coordinates, charge, ms):
    return _uhf(symbols, coordinates, charge, ms).E


def _uhf_gradient(symbols, coordinates, charge, ms):
    return _uhf(symbols, coordinates, charge, ms).gradient()


def _h2_system(**kwargs):
    return System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="sto-3g",
        auxiliary_basis_set="def2-universal-JKFIT",
        unit="bohr",
        **kwargs,
    )


def test_uhf_gradient_water_cation_finite_difference_and_translation():
    """Check an open-shell polyatomic UHF gradient against finite differences."""
    symbols = ["O", "H", "H"]
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8],
            [1.6, 0.0, 0.0],
        ]
    )

    gradient = _uhf_gradient(symbols, coordinates, charge=1, ms=0.5)

    for atom in range(3):
        for cart in range(3):
            numerical = four_point_central_difference_gradient_component(
                _uhf_energy, symbols, coordinates, atom, cart, charge=1, ms=0.5
            )
            assert gradient[atom, cart] == pytest.approx(numerical, abs=1.0e-8)

    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_uhf_gradient_h2_cation_six_point_finite_difference():
    """Check a small one-electron UHF gradient with the six-point stencil."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
    gradient = _uhf_gradient(symbols, coordinates, charge=1, ms=0.5)

    for atom in range(2):
        for cart in range(3):
            numerical = six_point_central_difference_gradient_component(
                _uhf_energy, symbols, coordinates, atom, cart, charge=1, ms=0.5
            )
            assert gradient[atom, cart] == pytest.approx(numerical, abs=1.0e-8)


def test_uhf_gradient_n2_cation_quartet_finite_difference_and_translation():
    """Check a higher-spin UHF gradient and translational invariance."""
    symbols = ["N", "N"]
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.1],
        ]
    )

    gradient = _uhf_gradient(symbols, coordinates, charge=1, ms=1.5)

    for atom in range(2):
        for cart in range(3):
            numerical = four_point_central_difference_gradient_component(
                _uhf_energy, symbols, coordinates, atom, cart, charge=1, ms=1.5
            )
            assert gradient[atom, cart] == pytest.approx(numerical, abs=1.0e-8)

    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test_uhf_gradient_auto_runs_and_reuses_executed_object():
    """Ensure UHF.gradient() runs SCF on demand and is repeatable afterward."""
    system = _h2_system()
    uhf = UHF(charge=1, ms=0.5, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)

    assert not uhf.executed
    gradient1 = uhf.gradient()
    energy1 = uhf.E

    assert uhf.executed
    gradient2 = uhf.gradient()

    assert uhf.E == pytest.approx(energy1)
    assert gradient1 == pytest.approx(gradient2, abs=1.0e-12)
    assert gradient1.shape == (system.natoms, 3)


def test_uhf_gradient_rejects_cholesky_tei():
    """Reject the Cholesky ERI path, which has no derivative backend yet."""
    system = System(
        xyz="H 0 0 0\nH 0 0 1.7",
        basis_set="sto-3g",
        cholesky_tei=True,
        unit="bohr",
    )
    uhf = UHF(charge=1, ms=0.5, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    uhf.run()

    with pytest.raises(NotImplementedError, match="density fitting"):
        uhf.gradient()


def test_uhf_gradient_with_df_ortho_rtol():
    """Test the DF metric orthogonalization path in the UHF gradient.

    When ``df_ortho_rtol`` is set, the DF metric inverse is built from a truncated
    eigenspace rather than the default full Cholesky solve. This is a test
    for that branch of ``build_metric_inverted_three_center`` inside the gradient
    assembly; the finite-difference tests cover the numerical gradient formula.
    """
    system = _h2_system(df_ortho_rtol=1.0e-8)
    uhf = UHF(charge=1, ms=0.5, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)

    gradient = uhf.gradient()

    assert gradient.shape == (system.natoms, 3)


def test_uhf_gradient_closed_shell_matches_rhf_gradient():
    """Verify the spin-paired UHF gradient reduces to the RHF gradient."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]])
    system = make_test_system(symbols, coordinates)

    rhf = RHF(charge=0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)
    uhf = UHF(charge=0, ms=0.0, e_tol=1.0e-12, d_tol=1.0e-10, maxiter=100)(system)

    rhf_gradient = rhf.gradient()
    uhf_gradient = uhf.gradient()

    assert uhf.E == pytest.approx(rhf.E, abs=1.0e-12)
    assert uhf_gradient == pytest.approx(rhf_gradient, abs=1.0e-10)


@pytest.mark.parametrize(
    "system_options",
    [
        {"use_gaussian_charges": True},
        {"x2c_type": "sf"},
        {"x2c_type": "sf", "use_gaussian_charges": True},
    ],
)
def test_uhf_gradient_nuclear_model_and_x2c_finite_difference(system_options):
    def energy(distance):
        system = System(
            xyz=f"H 0 0 0\nH 0 0 {distance:.12f}",
            basis_set="sto-3g",
            auxiliary_basis_set="def2-universal-JKFIT",
            unit="bohr",
            **system_options,
        )
        return UHF(charge=1, ms=0.5, e_tol=1.0e-12, d_tol=1.0e-10)(system).run().E

    system = _h2_system(**system_options)
    gradient = UHF(charge=1, ms=0.5, e_tol=1.0e-12, d_tol=1.0e-10)(system).gradient()
    step = 1.0e-3
    numerical = (
        -energy(1.7 + 2.0 * step)
        + 8.0 * energy(1.7 + step)
        - 8.0 * energy(1.7 - step)
        + energy(1.7 - 2.0 * step)
    ) / (12.0 * step)

    assert gradient[1, 2] == pytest.approx(numerical, abs=1.0e-8)
