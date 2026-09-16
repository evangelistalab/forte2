# Reference energies come from PySCF 2.13.1 (`pyscf.scf.dhf.DHF(mol).density_fit(...)`,
# Kramers-unrestricted, full Dirac-Coulomb including (SS|SS)), run with
# `pyscf.lib.param.LIGHT_SPEED` set to Forte2's value and with the orbital and
# auxiliary bases loaded from Forte2's own bundled BSE JSON files, so that the two
# codes see bit-identical basis data. See `tests/scf/dhf_references.py`.

import numpy as np
import pytest

from forte2 import System, X2CParams
from forte2.helpers.comparisons import approx, approx_abs
from forte2.integrals import LIBCINT_AVAILABLE
from forte2.scf import DHF, RHF, UHF
from forte2.x2c.x2c import LIGHT_SPEED

pytestmark = pytest.mark.skipif(
    not LIBCINT_AVAILABLE, reason="Dirac-Hartree-Fock requires libcint"
)

AUX = "def2-universal-jkfit"
H2O_XYZ = """
O 0.000000000000 0.000000000000 -0.061664597388
H 0.000000000000 -0.711620616369 0.489330954643
H 0.000000000000 0.711620616369 0.489330954643
"""


def _system(xyz, basis_set="cc-pvdz", **kwargs):
    return System(xyz=xyz, basis_set=basis_set, auxiliary_basis_set=AUX, **kwargs)


@pytest.mark.parametrize(
    "label,xyz,e_ref",
    [
        ("Ne", "Ne 0 0 0", -128.631796120994),
        ("Ar", "Ar 0 0 0", -528.632515255352),
        ("Kr", "Kr 0 0 0", -2786.365052825994),
        ("HF", "H 0 0 0\nF 0 0 0.91693", -100.110336848564),
        ("H2O", H2O_XYZ, -76.076569184617),
    ],
)
def test_dhf_energy(label, xyz, e_ref):
    """Density-fitted Dirac-Coulomb HF energies against PySCF's DF4C reference."""
    scf = DHF(charge=0)(_system(xyz))
    scf.run()
    assert scf.converged
    assert scf.E == approx(e_ref)


def test_dhf_orbital_structure():
    """Ne: Kramers pairs stay degenerate and the 2p spin-orbit splitting is right."""
    scf = DHF(charge=0)(_system("Ne 0 0 0"))
    scf.run()

    # The negative-energy branch is separated from the electronic states by ~2c^2.
    eps = scf.eps[0]
    n_neg = int(np.sum(eps < -(LIGHT_SPEED**2)))
    assert n_neg == scf.n_negative
    assert eps[n_neg - 1] < -(LIGHT_SPEED**2) < eps[n_neg]

    occ = eps[n_neg : n_neg + 10]
    # Time-reversal symmetry makes every electronic level a degenerate Kramers pair.
    assert np.allclose(occ[0::2], occ[1::2], atol=1e-9)
    # 1s1/2, 2s1/2, 2p1/2, then the fourfold 2p3/2 level.
    assert occ[0] == approx(-32.817966241)
    assert occ[2] == approx(-1.924102759)
    assert occ[4] == approx(-0.834372330)
    assert occ[6] == approx(-0.830262502)
    assert occ[6] - occ[4] == approx_abs(0.004109828, 1e-8)


def test_dhf_nonrelativistic_limit():
    """Scaling up c must drive DHF onto RHF as 1/c^2.

    Checking the scaling at two points is sharper than comparing one scaled energy
    against RHF, and needs no external reference. Scaling much beyond 100x is not
    useful: the four-component eigenvalue spread grows as c^2, and the SCF stops
    reaching the default density threshold.
    """
    rhf = RHF(charge=0)(_system("Ne 0 0 0"))
    rhf.run()

    deviations = {}
    for factor in (10, 30):
        dhf = DHF(charge=0, c_light=factor * LIGHT_SPEED)(_system("Ne 0 0 0"))
        dhf.run()
        assert dhf.converged
        deviations[factor] = dhf.E - rhf.E

    assert deviations[30] == approx_abs(0.0, 2e-4)
    assert deviations[10] < 0.0
    assert deviations[30] < 0.0
    assert deviations[10] / deviations[30] == approx_abs(9.0, 0.01)


def test_dhf_open_shell_nonrelativistic_limit():
    """The aufbau path over an odd electron count takes the same limit onto UHF.

    Lithium rather than a p-block atom: a degenerate open p shell leaves the hole
    placement ambiguous, and the two methods need not pick the same one.
    """
    uhf = UHF(charge=0, ms=0.5)(_system("Li 0 0 0"))
    uhf.run()

    deviations = {}
    for factor in (10, 30):
        dhf = DHF(charge=0, c_light=factor * LIGHT_SPEED)(_system("Li 0 0 0"))
        dhf.run()
        assert dhf.converged
        deviations[factor] = dhf.E - uhf.E

    assert deviations[30] == approx_abs(0.0, 2e-5)
    assert deviations[10] < 0.0
    assert deviations[30] < 0.0
    assert deviations[10] / deviations[30] == approx_abs(9.0, 0.01)


def test_dhf_gaussian_nuclear_charges():
    """Finite (Gaussian) nuclei lower the magnitude of the Dirac-Coulomb energy."""
    scf = DHF(charge=0)(_system("Ne 0 0 0", use_gaussian_charges=True))
    scf.run()
    # PySCF with `nucmod=1` (Visscher-Dyall); the two codes use slightly different
    # nuclear radii, hence the loosened tolerance.
    assert scf.E == approx_abs(-128.631760361144, 1e-7)


def test_dhf_hcore_matches_x2c_dirac_matrix():
    """The four-component core Hamiltonian agrees with the one X2C decouples.

    `X2CHelper` builds the same matrix Dirac equation in its decontracted basis, so
    the electronic eigenvalues of the two builders must coincide exactly.
    """
    system = System(
        xyz="Ne 0 0 0",
        basis_set="sto-3g",
        auxiliary_basis_set=AUX,
        x2c=X2CParams(x2c_type="so", x2c_model="1e"),
        minao_basis_set=None,
    )
    helper = system.x2c_helper
    eps_ref, _ = helper._solve_dirac_eq(*helper._get_integrals())

    from forte2.scf.dirac import dirac_hcore, dirac_orthogonalizer, dirac_overlap

    h = dirac_hcore(system, basis=helper.xbasis)
    S = dirac_overlap(system, basis=helper.xbasis)
    X, _, _ = dirac_orthogonalizer(system, basis=helper.xbasis)
    eps = np.linalg.eigvalsh(X.conj().T @ h @ X)

    assert np.allclose(np.sort(eps), np.sort(eps_ref), atol=1e-8)
    assert np.allclose(h, h.conj().T)
    assert np.allclose(S, S.conj().T)


def test_dhf_guess_types_agree():
    """The SAP and bare-core Dirac guesses converge to the same solution."""
    sap = DHF(charge=0, guess_type="minao")(_system("Ar 0 0 0"))
    sap.run()
    core = DHF(charge=0, guess_type="hcore")(_system("Ar 0 0 0"))
    core.run()
    assert sap.E == approx(core.E)
    assert sap.iter < core.iter
