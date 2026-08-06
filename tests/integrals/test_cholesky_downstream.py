"""
Sub-phase 3.4 validation for the production Cholesky ERI backends (``cholesky_tei`` in
``{"otf", "pivoted"}``).

The oracle gate (``test_cholesky_reference_oracle.py``) and the per-driver hardening tests pin the
decomposition itself, but only on small, dense-ERI-formable systems (up to cc-pVDZ / d functions).
This module closes the remaining validation gaps needed for the feature to be production-ready:

* **Downstream methods.** A full MCSCF (``MCOptimizer``) and a DSRG-MRPT2 run must give the same
  energy with ``cholesky_tei="otf"`` and ``"pivoted"`` as with the dense-ERI Cholesky path
  (``"naive"``). ``"naive"`` -- not density fitting -- is the right yardstick: all three decompose
  the *same* exact operator, so they agree to the CD tolerance, whereas DF is a different
  approximation and would only agree to its own (looser) error.
* **High angular momentum.** cc-pVTZ (f) and cc-pVQZ (g) drive the Schwarz-screened four-center
  kernel and both decomposition drivers past the d-limited oracle. No standard basis reaches
  libcint's ``l > 6`` gate, so both stay on the libint2 backend; this is the meaningful high-l
  coverage the feature can exercise.
* **Scale / low-rank footprint.** On a medium system (water dimer, cc-pVDZ) both drivers must select
  far fewer Cholesky vectors than the full ``nbf**2`` AO-pair count and still reconstruct the ERI --
  the low-rank structure the whole approach relies on.

Gradients are intentionally not covered here: the analytic DF-gradient path hard-rejects *every*
``cholesky_tei`` mode. That rejection is pinned for all modes in ``test_rhf_gradient.py`` and
``test_casscf_gradient.py``.
"""

import numpy as np
import pytest

from forte2 import System, RHF, MCOptimizer, State, CISolver, integrals
from forte2.dsrg import DSRG_MRPT2
from forte2.integrals.cholesky import cholesky_otf, cholesky_pivoted

# The exact-operator Cholesky modes: all three reproduce the true ERI, so downstream energies must
# agree to the CD tolerance regardless of which one built the B tensor.
EXACT_MODES = ("naive", "otf", "pivoted")

WATER = """
O 0.000000000000 0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369 0.489330954643
H 0.000000000000 0.711620616369 0.489330954643
"""

N2 = """
N 0.0 0.0 0.0
N 0.0 0.0 1.120
"""

# Water dimer: ~48 cc-pVDZ basis functions -- still small enough to form the dense (mn|rs) as ground
# truth, but large enough that the Cholesky rank is a small fraction of nbf**2.
WATER_DIMER = """
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
"""


def _recon_bound(tol):
    """Max-abs reconstruction bound at threshold ``tol`` (same convention as the oracle gate)."""
    return max(1e-9, 32.0 * tol)


# ---------------------------------------------------------------------------
# Downstream methods: otf/pivoted must match the dense-ERI Cholesky ("naive") path.
# ---------------------------------------------------------------------------
def test_mcscf_energy_agrees_across_cholesky_modes():
    """CASSCF(6,6)/cc-pVDZ on N2 must give the same energy for naive/otf/pivoted.

    All three decompose the exact ERI, so the SCF and MCSCF energies must agree to well within the
    ``cholesky_tol=1e-10`` decomposition accuracy -- a genuine end-to-end check that the on-the-fly
    and two-step B tensors feed MCSCF identically to the dense reference.
    """
    energies = {}
    for mode in EXACT_MODES:
        system = System(
            xyz=N2,
            basis_set="cc-pVDZ",
            cholesky_tei=mode,
            cholesky_tol=1e-10,
            symmetry=True,
        )
        rhf = RHF(charge=0, e_tol=1e-12)(system)
        ci_solver = CISolver(
            State(nel=14, multiplicity=1, ms=0.0),
            core_orbitals=[0, 1, 2, 3],
            active_orbitals=[4, 5, 6, 7, 8, 9],
        )
        mc = MCOptimizer(ci_solver, e_tol=1e-9)(rhf)
        mc.run()
        energies[mode] = (rhf.E, mc.E)

    # Anchor the reference path to its known value, then require the on-the-fly modes to match it.
    assert energies["naive"][0] == pytest.approx(-108.949591958787, abs=1e-8)
    assert energies["naive"][1] == pytest.approx(-109.090719613072, abs=1e-8)
    for mode in ("otf", "pivoted"):
        assert energies[mode][0] == pytest.approx(energies["naive"][0], abs=1e-8)
        assert energies[mode][1] == pytest.approx(energies["naive"][1], abs=1e-8)


def test_dsrg_mrpt2_energy_agrees_across_cholesky_modes():
    """DSRG-MRPT2 on H2/sto-6g must give the same energy for naive/otf/pivoted.

    A minimal all-active chain (RHF -> CISolver -> MCOptimizer -> DSRG_MRPT2) exercises the whole
    downstream consumer of the active-space integral triple; the correlated energy must not depend
    on which exact-ERI Cholesky mode produced the integrals.
    """
    energies = {}
    for mode in EXACT_MODES:
        system = System(
            xyz="H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
            basis_set="sto-6g",
            cholesky_tei=mode,
            cholesky_tol=1e-10,
        )
        rhf = RHF(charge=0, e_tol=1e-12)(system)
        ci_solver = CISolver(
            State(nel=2, multiplicity=1, ms=0.0), active_orbitals=[0, 1]
        )
        mc = MCOptimizer(ci_solver, e_tol=1e-9)(rhf)
        pt = DSRG_MRPT2(flow_param=0.5)(mc)
        pt.run()
        energies[mode] = pt.E_dsrg

    assert energies["naive"] == pytest.approx(-1.145939810, abs=1e-8)
    for mode in ("otf", "pivoted"):
        assert energies[mode] == pytest.approx(energies["naive"], abs=1e-8)


# ---------------------------------------------------------------------------
# High angular momentum: exercise the screened kernel + both drivers past the d-limited oracle.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("driver", [cholesky_otf, cholesky_pivoted])
def test_high_l_ccpvtz_reconstructs(driver):
    """cc-pVTZ (f functions) reconstructs the dense ERI for both drivers.

    The oracle gate tops out at cc-pVDZ (d). This drives the Schwarz-screened four-center primitive
    and the significant-set / draining logic on f shells, where screening and block layout are
    genuinely different from the d case.
    """
    tol = 1e-8
    system = System(xyz="Ne 0 0 0", basis_set="cc-pVTZ")
    nbf = system.nbf
    B, naux = driver(system, tol)
    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)
    gram = B.reshape(naux, nbf * nbf)
    gram = gram.T @ gram
    assert np.max(np.abs(gram - M)) <= _recon_bound(tol)


@pytest.mark.slow
@pytest.mark.parametrize("driver", [cholesky_otf, cholesky_pivoted])
def test_high_l_ccpvqz_reconstructs(driver):
    """cc-pVQZ (g functions) reconstructs the dense ERI for both drivers (slow: g-shell kernel)."""
    tol = 1e-8
    system = System(xyz="Ne 0 0 0", basis_set="cc-pVQZ")
    nbf = system.nbf
    B, naux = driver(system, tol)
    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)
    gram = B.reshape(naux, nbf * nbf)
    gram = gram.T @ gram
    assert np.max(np.abs(gram - M)) <= _recon_bound(tol)


# ---------------------------------------------------------------------------
# Scale / low-rank footprint on a medium system.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("driver", [cholesky_otf, cholesky_pivoted])
def test_medium_system_low_rank_and_reconstructs(driver):
    """On the water dimer (cc-pVDZ, ~48 bf) the Cholesky rank is a small fraction of nbf**2.

    The whole approach is worthwhile only if the number of selected vectors stays well below the
    full AO-pair count; here naux/nbf**2 is well under a half. The decomposition must also still
    reconstruct the dense ERI to tolerance at this larger size.
    """
    tol = 1e-8
    system = System(xyz=WATER_DIMER, basis_set="cc-pVDZ")
    nbf = system.nbf
    B, naux = driver(system, tol)

    # Low-rank: far fewer vectors than the nbf**2 AO pairs (measured ratio ~0.24).
    assert naux < 0.5 * nbf * nbf

    M = integrals.coulomb_4c(system).reshape((nbf * nbf,) * 2)
    gram = B.reshape(naux, nbf * nbf)
    gram = gram.T @ gram
    assert np.max(np.abs(gram - M)) <= _recon_bound(tol)
