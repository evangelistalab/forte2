# Four-component CI over the positive-energy (no-pair) spinors of a DHF reference.
#
# The carbon reference energies were reproduced independently by PySCF's
# `pyscf.fci.fci_dhf_slow.kernel`, fed the same active-space integrals converted
# from Forte2's physicist ordering to PySCF's chemist ordering
# (`eri = V.transpose(0, 2, 1, 3)`). See `tests/scf/dhf_references.py`.

import numpy as np
import pytest

from forte2 import CI, DHF, MCOptimizer, RelCISolver, System
from forte2.helpers.comparisons import approx
from forte2.integrals import LIBCINT_AVAILABLE

pytestmark = pytest.mark.skipif(
    not LIBCINT_AVAILABLE, reason="Dirac-Hartree-Fock requires libcint"
)


def _system(xyz):
    return System(
        xyz=xyz,
        basis_set="decon-cc-pvdz",
        auxiliary_basis_set="def2-universal-jkfit",
    )


@pytest.mark.parametrize(
    "core,active",
    [
        pytest.param(10, 2, id="frozen-core"),
        pytest.param(0, 10, id="all-active"),
    ],
)
def test_dhf_casci_reproduces_dhf_energy(core, active):
    """A CAS holding no correlation must return the DHF energy exactly.

    The two parametrizations exercise different code: with every electron frozen
    the energy comes from the core Fock contraction, while with every electron
    active it comes from the transformed two-electron tensor.
    """
    scf = DHF(charge=0)(_system("Ne 0 0 0"))
    scf.run()
    ci = CI(RelCISolver(nel=10, core_orbitals=core, active_orbitals=active))(scf)
    ci.run()
    assert ci.E_ci[0].real == approx(scf.E)
    assert abs(ci.E_ci[0].imag) < 1e-10


def test_dhf_mos_span_the_electronic_branch():
    """DHF hands downstream methods the no-pair space, not the whole spectrum."""
    scf = DHF(charge=0)(_system("Ne 0 0 0"))
    scf.run()
    assert scf.mos.C[0].shape == (4 * scf.nbf, scf.n_positive)
    assert scf.C[0].shape == (4 * scf.nbf, scf.n_positive + scf.n_negative)
    assert len(scf.eps_electronic) == scf.n_positive
    # The electronic branch is exactly what MOSpace sizes itself against.
    assert scf.n_positive == 2 * scf.system.nmo


def test_dhf_casci_orbital_rotation_invariance():
    """Mixing orbitals inside a subspace leaves the CASCI energy alone."""
    scf = DHF(charge=0)(_system("C 0 0 0"))
    scf.run()
    reference = CI(RelCISolver(nel=6, core_orbitals=2, active_orbitals=8, nroots=9))(
        scf
    )
    reference.run()

    rng = np.random.default_rng(20260916)
    C = scf.mos.C[0].copy()
    for lo, hi in ((0, 2), (2, 10)):
        n = hi - lo
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        A -= A.conj().T
        C[:, lo:hi] = C[:, lo:hi] @ np.linalg.qr(np.eye(n) + 0.1 * A)[0]
    scf.mos.C[0] = C

    rotated = CI(RelCISolver(nel=6, core_orbitals=2, active_orbitals=8, nroots=9))(scf)
    rotated.run()
    assert np.allclose(
        np.array(rotated.E_ci).real, np.array(reference.E_ci).real, atol=1e-9
    )


def test_dhf_casci_carbon_matches_pyscf():
    """Nine roots of the carbon 2s2p CAS, cross-checked against PySCF's DHF FCI."""
    scf = DHF(charge=2)(_system("C 0 0 0"))
    scf.run()
    mc = MCOptimizer(RelCISolver(nel=6, core_orbitals=2, active_orbitals=8, nroots=9))(
        scf
    )
    mc.run()
    roots = np.array(mc.E_ci).real
    expected = np.array(
        [-37.7161568799]
        + [-37.7160794557] * 3
        + [-37.7159257927] * 4
        + [-37.7159257926]
    )
    assert np.allclose(roots, expected, atol=5e-8)
