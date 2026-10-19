import numpy as np
import pytest

from forte2 import GHF, MCOptimizer, RelCISolver, RelState, System, X2CParams
from forte2.data.atom_data import EH_TO_WN, LIGHT_SPEED
from forte2.helpers.comparisons import approx


def _h3_sa_casscf(light_speed):
    """SO-X2C SA-CASSCF(3e, 6 spinors) of equilateral H3 (side 1 Angstrom),
    averaged over the four 2E' states."""
    radius = 1.0 / np.sqrt(3.0)
    angles = np.pi / 2 + 2 * np.pi / 3 * np.arange(3)
    xyz = "\n".join(
        f"H {radius * np.cos(t):.12f} {radius * np.sin(t):.12f} 0.0" for t in angles
    )
    system = System(
        xyz=xyz,
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=X2CParams(x2c_type="so", light_speed=light_speed),
    )
    scf = GHF(charge=0)(system)
    ci = RelCISolver(
        RelState(charge=0, system=system),
        nroots=4,
        core_orbitals=0,
        active_orbitals=6,
    )
    mc = MCOptimizer(
        ci,
        e_tol=1e-12,
        g_tol=1e-9,
        final_orbitals="original",
    )(scf)
    mc.run()
    return mc


def test_h3_sa_casscf_spin_orbit_gap_scales_with_light_speed():
    # In the nonrel regime, D3h H3 is a conical intersection of the 2E' pair.
    # Spin-orbit coupling splits the four states into two Kramers pairs, with a
    # gap that scales to leading order as 1 / c**2.
    eavg_ref = {1: -1.5141946972, 5: -1.5144359004, 10: -1.5151699148}
    gaps = {}
    for scale, eavg in eavg_ref.items():
        mc = _h3_sa_casscf(LIGHT_SPEED / scale)
        E = np.asarray(mc.E_ci).real
        assert mc.E_avg.real == approx(eavg)
        # degeneracy within kramers pairs
        assert E[1] == pytest.approx(E[0], abs=1e-10)
        assert E[3] == pytest.approx(E[2], abs=1e-10)
        gaps[scale] = (E[2] - E[1]) * EH_TO_WN

    assert gaps[1] == pytest.approx(1.5749, abs=1e-3)
    for scale in (5, 10):
        assert gaps[scale] / scale**2 == pytest.approx(gaps[1], rel=2e-3)
