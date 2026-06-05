"""
Tests for the spectral (spatial) localizer.

The checks below validate the implementation against cases whose answers are
known analytically or from textbook topological band theory, so correctness does
not rely on any external reference:

* a single isolated site (analytic localizer gap),
* atomic-limit (decoupled) insulators in 1D and 2D (centers sit on the sites),
* the Qi-Wu-Zhang Chern insulator (local index = Chern number = +/-1 in the
  topological phase, 0 in the trivial phase),
* a molecular H2 example (the electron pair localizes at the bond midpoint).
"""

import numpy as np
import pytest

from forte2 import SpectralLocalizer, System, RHF


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _qwz_hamiltonian(lx, ly, mass):
    """Real-space Qi-Wu-Zhang Hamiltonian on an open lx-by-ly lattice.

    Two orbitals per site, basis ordered site-major with site index ix*ly+iy.
    The lower-band Chern number is -sgn(mass) for 0 < |mass| < 2 and 0 otherwise.
    """
    nsites = lx * ly
    H = np.zeros((2 * nsites, 2 * nsites), dtype=complex)
    tx = 0.5 * _SZ - 0.5j * _SX
    ty = 0.5 * _SZ - 0.5j * _SY

    def block(s):
        return slice(2 * s, 2 * s + 2)

    def idx(ix, iy):
        return ix * ly + iy

    for ix in range(lx):
        for iy in range(ly):
            s = idx(ix, iy)
            H[block(s), block(s)] += mass * _SZ
            if ix + 1 < lx:
                t = idx(ix + 1, iy)
                H[block(t), block(s)] += tx
                H[block(s), block(t)] += tx.conj().T
            if iy + 1 < ly:
                t = idx(ix, iy + 1)
                H[block(t), block(s)] += ty
                H[block(s), block(t)] += ty.conj().T

    coords = np.array([[ix, iy] for ix in range(lx) for iy in range(ly)], dtype=float)
    return H, coords


# --------------------------------------------------------------------------- #
# Analytic single-site checks
# --------------------------------------------------------------------------- #
def test_single_site_gap_closes_at_site():
    # One site at x = 2 with onsite energy 0. For N=1, the 1D localizer is
    # [[ -E, k(2-x) ], [ k(2-x), E ]], with eigenvalues +/- sqrt(E^2+k^2(2-x)^2).
    site = 2.0
    loc = SpectralLocalizer(np.array([[0.0]]), [np.array([[site]])], kappa=1.0)

    assert loc.localizer_gap([site], energy=0.0) == pytest.approx(0.0, abs=1e-12)
    # Away from the site the gap is exactly kappa*|x - site|.
    assert loc.localizer_gap([0.0], energy=0.0) == pytest.approx(2.0)
    assert loc.localizer_gap([5.0], energy=0.0) == pytest.approx(3.0)


def test_single_site_is_hermitian():
    loc = SpectralLocalizer(np.array([[0.3]]), [np.array([[1.0]])], kappa=0.7)
    L = loc.localizer([0.5], energy=0.1)
    assert np.allclose(L, L.conj().T)


# --------------------------------------------------------------------------- #
# Atomic-limit insulators (centers on the sites)
# --------------------------------------------------------------------------- #
def test_atomic_limit_1d_centers():
    sites = np.array([0.0, 3.0, 7.0])
    H = np.zeros((3, 3))  # decoupled sites
    loc = SpectralLocalizer.from_lattice(H, sites, kappa=1.0)
    centers, gaps = loc.find_centers(energy=0.0, ngrid=29)
    assert centers.shape == (3, 1)
    found = np.sort(centers.ravel())
    assert found == pytest.approx(sites, abs=1e-3)
    assert np.allclose(gaps, 0.0, atol=1e-3)


def test_atomic_limit_2d_centers():
    H, coords = _qwz_hamiltonian(3, 3, mass=0.0)  # only used for coords
    # Build a truly decoupled 2D lattice (atomic limit): diagonal onsite only.
    nsites = coords.shape[0]
    Hdiag = np.zeros((nsites, nsites))
    loc = SpectralLocalizer.from_lattice(Hdiag, coords, kappa=1.0)
    centers, gaps = loc.find_centers(energy=0.0, ngrid=13)
    assert centers.shape[0] == nsites
    # Every site coordinate should be recovered.
    for c in coords:
        assert np.min(np.linalg.norm(centers - c, axis=1)) == pytest.approx(0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Chern insulator: local index = Chern number
# --------------------------------------------------------------------------- #
def test_qwz_local_chern_marker():
    lx = ly = 11
    center_site = (lx // 2) * ly + (ly // 2)
    probe = np.array([lx // 2, ly // 2], dtype=float)

    # Topological phase: |C| = 1.
    H_topo, coords = _qwz_hamiltonian(lx, ly, mass=-1.0)
    loc_topo = SpectralLocalizer.from_lattice(
        H_topo, coords, orbitals_per_site=2, kappa=1.0
    )
    idx_topo = loc_topo.local_index(probe, energy=0.0)
    assert abs(idx_topo) == 1
    # The bulk localizer gap must be open for the index to be meaningful.
    assert loc_topo.localizer_gap(probe, energy=0.0) > 1e-2

    # Trivial phase: C = 0.
    H_triv, _ = _qwz_hamiltonian(lx, ly, mass=-3.0)
    loc_triv = SpectralLocalizer.from_lattice(
        H_triv, coords, orbitals_per_site=2, kappa=1.0
    )
    assert loc_triv.local_index(probe, energy=0.0) == 0


def test_qwz_chern_sign_flips_with_mass():
    lx = ly = 9
    probe = np.array([lx // 2, ly // 2], dtype=float)
    H_pos, coords = _qwz_hamiltonian(lx, ly, mass=-1.0)
    H_neg, _ = _qwz_hamiltonian(lx, ly, mass=+1.0)
    loc_pos = SpectralLocalizer.from_lattice(H_pos, coords, orbitals_per_site=2, kappa=1.0)
    loc_neg = SpectralLocalizer.from_lattice(H_neg, coords, orbitals_per_site=2, kappa=1.0)
    c_pos = loc_pos.local_index(probe, energy=0.0)
    c_neg = loc_neg.local_index(probe, energy=0.0)
    assert abs(c_pos) == 1 and abs(c_neg) == 1
    assert c_pos == -c_neg


# --------------------------------------------------------------------------- #
# Model-system constructor
# --------------------------------------------------------------------------- #
def test_from_model_system_hubbard_coords():
    from forte2 import HubbardModel

    model = HubbardModel(t=1.0, U=0.0, nsites=4)
    loc = SpectralLocalizer.from_model_system(model, kappa=1.0)
    assert loc.ndim == 1
    assert loc.nbasis == 4
    # The position operator should be diag(0, 1, 2, 3).
    assert np.allclose(np.diag(loc.positions[0]).real, [0, 1, 2, 3])


# --------------------------------------------------------------------------- #
# Molecular example: H2 electron pair localizes at the bond midpoint
# --------------------------------------------------------------------------- #
def test_h2_molecular_constructor():
    bond = 0.74  # angstrom
    xyz = f"H 0.0 0.0 0.0\nH 0.0 0.0 {bond}"
    system = System(xyz=xyz, basis_set="sto-3g", cholesky_tei=True)
    scf = RHF(charge=0)(system)
    scf.run()

    loc = SpectralLocalizer.from_scf(scf, kappa=1.0)
    # A molecular localizer lives in 3D with one operator per Cartesian axis,
    # and the basis is the set of MOs.
    assert loc.ndim == 3
    assert loc.nbasis == 2

    # The position operators feeding the localizer are the dipole integrals in the
    # MO basis. The occupied (bonding) orbital's center -- the maximally localized
    # Wannier/Boys center for this single-pair system -- is the bond midpoint by
    # symmetry. The integrals are in atomic units (bohr).
    bohr = 1.8897259886
    z_mid = 0.5 * bond * bohr
    z_op = loc.positions[2]
    assert z_op[0, 0].real == pytest.approx(z_mid, abs=1e-6)

    # The localizer is Hermitian with real spectrum.
    L = loc.localizer([0.0, 0.0, z_mid], energy=loc.reference_energy)
    assert L.shape == (8, 8)  # 2 MOs x 4-component Clifford rep
    assert np.allclose(L, L.conj().T)
    assert np.allclose(loc.eigenvalues([0.0, 0.0, z_mid]).imag, 0.0, atol=1e-10)

    # The localized state is a normalized vector in the MO basis.
    state = loc.localized_state([0.0, 0.0, z_mid], energy=loc.reference_energy)
    assert state.shape == (2,)
    assert np.linalg.norm(state) == pytest.approx(1.0)
