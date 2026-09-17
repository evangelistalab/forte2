# State-averaged four-component CASSCF on the second-row p-block atoms, with a
# 2s2p active space (8 spinors) holding 3 to 7 electrons.
#
# The DHF reference is the 1s^2 2s^2 closed-shell cation rather than the neutral
# atom: aufbau DHF on a partly filled, degenerate p shell has an ill-defined
# occupied set and its density oscillates without converging. The cation is
# spherical and converges cleanly, and the state-averaged CASSCF re-optimizes the
# orbitals for the neutral anyway.
#
# Experimental levels are NIST Atomic Spectra Database ground-configuration fine
# structure, retrieved 2026-09-16.

import numpy as np
import pytest

from forte2 import DHF, GHF, MCOptimizer, RelCISolver, System
from forte2.data import EH_TO_WN
from forte2.helpers.comparisons import approx, approx_abs
from forte2.integrals import LIBCINT_AVAILABLE
from forte2.x2c.x2c import LIGHT_SPEED

pytestmark = pytest.mark.skipif(
    not LIBCINT_AVAILABLE, reason="Dirac-Hartree-Fock requires libcint"
)

# symbol -> (electrons, roots, state-averaged energy, [(level in cm-1, degeneracy)])
CASES = {
    "B": (5, 6, -24.565192168875, [(0.0, 2), (17.2416, 4)]),
    "C": (6, 9, -37.716002690043, [(0.0, 1), (16.9927, 3), (50.7178, 5)]),
    "N": (7, 14, -54.344058089206, [(0.0, 4), (23080.4536, 4), (23081.7308, 6)]),
    "O": (8, 9, -74.841634668781, [(0.0, 5), (152.7872, 3), (227.5320, 1)]),
    "F": (9, 6, -99.462731207078, [(0.0, 4), (406.5675, 2)]),
}

# NIST ASD, ground configuration, level in cm-1 with its J degeneracy.
NIST = {
    "B": [(0.0, 2), (15.287, 4)],  # 2P* 1/2, 3/2
    "C": [(0.0, 1), (16.4167, 3), (43.4135, 5)],  # 3P 0, 1, 2
    "N": [(0.0, 4), (19224.464, 6), (19233.177, 4)],  # 4S* 3/2; 2D* 5/2, 3/2
    "O": [(0.0, 5), (158.265, 3), (226.977, 1)],  # 3P 2, 1, 0 (inverted)
    "F": [(0.0, 4), (404.141, 2)],  # 2P* 3/2, 1/2 (inverted)
}


def _system(symbol, basis_set="decon-cc-pvdz"):
    return System(
        xyz=f"{symbol} 0 0 0",
        basis_set=basis_set,
        auxiliary_basis_set="def2-universal-jkfit",
    )


def _levels(energies, tol=0.5):
    """Collapse root energies into (level in cm-1, degeneracy) pairs."""
    rel = (np.asarray(energies).real - np.min(energies).real) * EH_TO_WN
    rel.sort()
    levels, start = [], 0
    for i in range(1, len(rel) + 1):
        if i == len(rel) or rel[i] - rel[start] > tol:
            levels.append((float(rel[start:i].mean()), i - start))
            start = i
    return levels


def _run(symbol):
    nel, nroots = CASES[symbol][0], CASES[symbol][1]
    # 1s^2 2s^2 closed-shell cation reference; the solver carries the neutral count.
    scf = DHF(charge=nel - 4)(_system(symbol))
    scf.run()
    mc = MCOptimizer(
        RelCISolver(nel=nel, core_orbitals=2, active_orbitals=8, nroots=nroots)
    )(scf)
    mc.run()
    return mc


@pytest.mark.parametrize("symbol", list(CASES))
def test_dhf_casscf_p_block_fine_structure(symbol):
    """Every J level of the ground configuration, with the right degeneracy."""
    nel, nroots, e_avg, expected = CASES[symbol]
    mc = _run(symbol)
    assert mc.converged
    assert mc.E.real == approx(e_avg)

    levels = _levels(mc.E_ci)
    assert sum(g for _, g in levels) == nroots
    assert [g for _, g in levels] == [g for _, g in expected]
    for (got, _), (want, _) in zip(levels, expected):
        assert got == approx_abs(want, 0.05)


@pytest.mark.parametrize("symbol", ["B", "C", "O", "F"])
def test_dhf_casscf_spin_orbit_splittings_track_experiment(symbol):
    """Fine-structure splittings land within 20% of the NIST levels.

    A 2s2p active space carries the one-electron spin-orbit coupling and the
    valence correlation that shapes it, but no dynamic correlation. The residual is
    the active space rather than the basis: moving to decon-cc-pvtz widens the gap
    (boron 1.13 to 1.18 of experiment, carbon 1.17 to 1.22), so the splittings are
    converging onto a CAS limit that overshoots. The ordering, including the
    inversion in oxygen and fluorine, has to be exact.
    """
    mc = _run(symbol)
    levels = _levels(mc.E_ci)
    reference = NIST[symbol]
    assert [g for _, g in levels] == [g for _, g in reference]
    for (got, _), (want, _) in zip(levels[1:], reference[1:]):
        assert 0.80 < got / want < 1.20


def test_dhf_casscf_nitrogen_2d_term():
    """Nitrogen's 4S* to 2D* term energy, and the near-degeneracy of the 2D pair.

    The 2D* fine structure is a second-order effect: the first-order spin-orbit
    matrix element vanishes within p^3, leaving a splitting of 8.7 cm-1 that a 2s2p
    active space puts at 1.3 cm-1 and in the opposite order. Only the magnitude is
    asserted here, not the ordering of the two 2D* levels.
    """
    mc = _run("N")
    levels = _levels(mc.E_ci)
    assert [g for _, g in levels] == [4, 4, 6]

    term = np.mean([levels[1][0], levels[2][0]])
    nist_term = np.mean([19224.464, 19233.177])
    assert 0.75 < term / nist_term < 1.25
    assert abs(levels[2][0] - levels[1][0]) < 10.0


def test_dhf_casscf_nonrelativistic_limit():
    """Scaling up c drives four-component CASSCF onto two-component CASSCF as 1/c^2.

    The reference is a GHF-based two-component CASSCF rather than a spin-restricted
    one: the four-component active space is built from unconstrained spinors, so it
    keeps the extra variational freedom a spinor CAS has even without spin-orbit
    coupling.
    """
    solver = lambda: RelCISolver(nel=4, active_orbitals=10)
    reference = MCOptimizer(solver())(GHF(charge=0)(_system("Be")))
    reference.run()
    assert reference.converged

    deviations = {}
    for factor in (10, 30):
        scf = DHF(charge=0, c_light=factor * LIGHT_SPEED)(_system("Be"))
        scf.run()
        mc = MCOptimizer(solver())(scf)
        mc.run()
        assert mc.converged
        deviations[factor] = mc.E.real - reference.E.real

    assert deviations[30] == approx_abs(0.0, 1e-5)
    assert deviations[10] < 0.0
    assert deviations[30] < 0.0
    assert deviations[10] / deviations[30] == approx_abs(9.0, 0.01)


def test_dhf_casscf_positronic_rotation_is_negligible_for_valence():
    """Optimizing electronic-positronic rotations leaves a valence CAS alone.

    Under the no-pair approximation the energy is a minimum with respect to
    rotations among electronic orbitals but a maximum with respect to rotations
    into the positronic branch, so releasing the latter can only raise the
    energy. Hoyer et al., J. Chem. Phys. 158, 044101 (2023), Table II, find the
    two schemes identical to every digit they quote for valence active spaces of
    Be through Ra; the effect is confined to deep-core correlation.
    """
    reference = _run("B")
    assert reference.converged

    scf = DHF(charge=1)(_system("B"))
    scf.run()
    relaxed = MCOptimizer(
        RelCISolver(nel=5, core_orbitals=2, active_orbitals=8, nroots=6),
        optimize_positronic=True,
    )(scf)
    relaxed.run()

    assert relaxed.converged
    assert relaxed.g_ep_rms < relaxed.g_tol
    # Freezing the maximization leaves the energy spuriously low, so releasing it
    # cannot lower the energy.
    assert relaxed.E.real >= reference.E.real - 1e-12
    assert relaxed.E.real == approx_abs(reference.E.real, 1e-9)


def test_dhf_casscf_positronic_rotation_needs_four_components():
    """Two-component references have no positronic branch to rotate into."""
    scf = GHF(charge=0)(_system("Be"))
    scf.run()
    mc = MCOptimizer(RelCISolver(nel=4, active_orbitals=10), optimize_positronic=True)(
        scf
    )
    with pytest.raises(ValueError, match="four-component"):
        mc.run()
