r"""
Spectral (spatial) localizer for locating electrons in finite systems.

This module implements the *spectral localizer* framework, a position-space tool
for probing the local topology of, and locating the electronic charge in, gapped
systems (insulators).  Given a single-particle Hamiltonian :math:`H` and a set of
commuting position operators :math:`X_1,\dots,X_d` (all expressed in a common
*orthonormal* one-particle basis), the spectral localizer at a probe point
:math:`\boldsymbol{\lambda}=(x_1,\dots,x_d)` and reference energy :math:`E` is the
Hermitian operator

.. math::

    L_{\boldsymbol{\lambda},E}(X, H)
        = (H - E)\otimes\Gamma_0
          + \kappa\sum_{j=1}^{d}(X_j - x_j)\otimes\Gamma_j ,

where :math:`\{\Gamma_0,\Gamma_1,\dots,\Gamma_d\}` is a Hermitian representation of
the Clifford algebra (:math:`\{\Gamma_a,\Gamma_b\}=2\delta_{ab}`) and
:math:`\kappa>0` balances the units of energy and length.

Two spectral quantities drive the framework:

* the **localizer gap** :math:`\mu(\boldsymbol{\lambda},E)=\min_i|\eta_i|`, where
  :math:`\eta_i` are the eigenvalues of :math:`L_{\boldsymbol{\lambda},E}`.  The gap
  measures how cleanly the probe point can be assigned a local topological index;
  it *closes* exactly at the locations of the electrons / Wannier centers.
* the **local index** :math:`\tfrac12\,\mathrm{sig}\,L_{\boldsymbol{\lambda},E}`,
  the half-signature of the localizer.  In two dimensions this is the local Chern
  number; quite generally it is the integer topological charge enclosed by the
  probe point.

Scanning the probe point at a reference energy inside the gap and tracking where
the localizer gap closes (equivalently, where the local index jumps) yields the
electron positions, generalizing the notion of Wannier centers to systems with
boundaries, defects, and disorder.

Notes
-----
This is an implementation of the established spectral-localizer formalism
(Loring; Loring & Schulz-Baldes; Cerjan & Benalcazar and co-workers) specialized
to the finite (molecular / open-lattice / tight-binding) systems that forte2
describes.  The low-dimensional Clifford representations used here reduce to the
familiar explicit forms, e.g. in 2D

.. math::

    L_{(x,y),E} =
    \begin{pmatrix} H-E & \kappa[(X-x)-i(Y-y)] \\
                    \kappa[(X-x)+i(Y-y)] & -(H-E) \end{pmatrix}.
"""

import numpy as np

__all__ = ["SpectralLocalizer"]

# Pauli matrices (complex).
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _clifford_gammas(d):
    r"""
    Return a Hermitian Clifford representation :math:`[\Gamma_0,\dots,\Gamma_d]`.

    The matrices are Hermitian, square to the identity, and mutually anticommute,
    :math:`\{\Gamma_a,\Gamma_b\}=2\delta_{ab}I`.  Element 0 multiplies the
    Hamiltonian; elements ``1..d`` multiply the ``d`` position operators.
    """
    if d == 1:
        # C_1: two anticommuting Hermitian matrices.
        return [_SIGMA_Z, _SIGMA_X]
    if d == 2:
        return [_SIGMA_Z, _SIGMA_X, _SIGMA_Y]
    if d == 3:
        # 4x4 Dirac-like representation.
        g0 = np.kron(_SIGMA_Z, _I2)
        g1 = np.kron(_SIGMA_X, _SIGMA_X)
        g2 = np.kron(_SIGMA_X, _SIGMA_Y)
        g3 = np.kron(_SIGMA_X, _SIGMA_Z)
        return [g0, g1, g2, g3]
    raise ValueError(f"Spatial dimension d={d} is not supported (use 1, 2, or 3).")


class SpectralLocalizer:
    r"""
    Spectral localizer for a finite single-particle system.

    Parameters
    ----------
    hamiltonian : ndarray
        The single-particle Hamiltonian :math:`H`, a Hermitian ``(N, N)`` matrix
        expressed in an **orthonormal** one-particle basis.
    positions : sequence of ndarray
        The position operators ``[X_1, ..., X_d]`` (``d`` in ``{1, 2, 3}``), each a
        Hermitian ``(N, N)`` matrix in the *same* orthonormal basis as
        ``hamiltonian``.  For a tight-binding / lattice model these are diagonal
        matrices holding the site coordinates; for a molecular system they are the
        dipole (position) integrals transformed to the MO basis.
    kappa : float, optional
        The scaling constant :math:`\kappa>0` balancing the energy and length
        scales.  If ``None`` (default), a heuristic value
        ``kappa = (energy spread of H) / (largest spatial spread)`` is used.

    Attributes
    ----------
    H : ndarray
        The Hamiltonian.
    positions : list of ndarray
        The position operators.
    ndim : int
        The spatial dimension ``d``.
    nbasis : int
        The one-particle basis dimension ``N``.
    kappa : float
        The scaling constant actually used.
    """

    def __init__(self, hamiltonian, positions, kappa=None):
        H = np.asarray(hamiltonian)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("`hamiltonian` must be a square (N, N) matrix.")
        positions = [np.asarray(X) for X in positions]
        d = len(positions)
        if d not in (1, 2, 3):
            raise ValueError("`positions` must contain 1, 2, or 3 operators.")
        for X in positions:
            if X.shape != H.shape:
                raise ValueError(
                    "Each position operator must match the shape of `hamiltonian`."
                )

        self.H = H.astype(complex)
        self.positions = [X.astype(complex) for X in positions]
        self.ndim = d
        self.nbasis = H.shape[0]
        self._gammas = _clifford_gammas(d)

        if kappa is None:
            kappa = self._default_kappa()
        if kappa <= 0:
            raise ValueError("`kappa` must be positive.")
        self.kappa = float(kappa)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _default_kappa(self):
        """Heuristic kappa balancing the energy and length scales."""
        evals = np.linalg.eigvalsh(self.H)
        energy_spread = float(evals.max() - evals.min())
        spatial_spread = 0.0
        for X in self.positions:
            xv = np.linalg.eigvalsh(X)
            spatial_spread = max(spatial_spread, float(xv.max() - xv.min()))
        if energy_spread <= 0 or spatial_spread <= 0:
            return 1.0
        return energy_spread / spatial_spread

    def localizer(self, point, energy=0.0):
        r"""
        Build the spectral localizer matrix :math:`L_{\boldsymbol{\lambda},E}`.

        Parameters
        ----------
        point : array-like
            The probe point :math:`\boldsymbol{\lambda}`, length ``ndim``.
        energy : float, optional
            The reference energy :math:`E` (default 0).

        Returns
        -------
        ndarray
            The Hermitian ``(sN, sN)`` localizer matrix, where ``s`` is the size of
            the Clifford representation (2 for ``d<=2``, 4 for ``d=3``).
        """
        point = np.atleast_1d(np.asarray(point, dtype=float))
        if point.shape[0] != self.ndim:
            raise ValueError(f"`point` must have length {self.ndim}.")
        eye = np.eye(self.nbasis, dtype=complex)
        # A_0 multiplies Gamma_0, A_j multiplies Gamma_j.
        blocks = [self.H - energy * eye]
        for j, X in enumerate(self.positions):
            blocks.append(self.kappa * (X - point[j] * eye))
        L = np.zeros((self.nbasis * len(self._gammas[0]),) * 2, dtype=complex)
        for A, gamma in zip(blocks, self._gammas):
            L += np.kron(A, gamma)
        return L

    # ------------------------------------------------------------------ #
    # Spectral quantities
    # ------------------------------------------------------------------ #
    def eigenvalues(self, point, energy=0.0):
        """Eigenvalues of the spectral localizer at ``point`` and ``energy``."""
        return np.linalg.eigvalsh(self.localizer(point, energy))

    def localizer_gap(self, point, energy=0.0):
        r"""
        The localizer gap :math:`\mu=\min_i|\eta_i|` at ``point`` and ``energy``.

        A small gap signals proximity to an electron / Wannier center (or, more
        generally, to a location where the local topological index changes).
        """
        return float(np.min(np.abs(self.eigenvalues(point, energy))))

    def local_index(self, point, energy=0.0, tol=1e-9):
        r"""
        The local topological index, the half-signature
        :math:`\tfrac12\,\mathrm{sig}\,L_{\boldsymbol{\lambda},E}`.

        In two dimensions this is the local Chern number.  The probe point should
        sit in an open localizer gap for the index to be well defined; a
        ``RuntimeWarning``-worthy non-integer result is rounded to the nearest
        integer.

        Parameters
        ----------
        point : array-like
            The probe point, length ``ndim``.
        energy : float, optional
            The reference energy (default 0).
        tol : float, optional
            Eigenvalues with magnitude below ``tol`` are treated as zero when
            counting the signature.

        Returns
        -------
        int
            The (rounded) half-signature.
        """
        evals = self.eigenvalues(point, energy)
        n_pos = int(np.count_nonzero(evals > tol))
        n_neg = int(np.count_nonzero(evals < -tol))
        signature = n_pos - n_neg
        return int(round(signature / 2))

    # ------------------------------------------------------------------ #
    # Locating electrons
    # ------------------------------------------------------------------ #
    def _default_bounds(self):
        """Bounding box (per dimension) from the spectra of the position ops."""
        bounds = []
        for X in self.positions:
            xv = np.linalg.eigvalsh(X)
            bounds.append((float(xv.min()), float(xv.max())))
        return bounds

    def gap_scan(self, energy=0.0, bounds=None, ngrid=21):
        r"""
        Evaluate the localizer gap on a regular grid of probe points.

        Parameters
        ----------
        energy : float, optional
            The reference energy :math:`E` (default 0), typically chosen inside the
            spectral gap of ``H`` (e.g. the Fermi level / middle of the HOMO-LUMO
            gap).
        bounds : sequence of (float, float), optional
            Per-dimension ``(min, max)`` ranges.  Defaults to the span of each
            position operator's spectrum.
        ngrid : int or sequence of int, optional
            Number of grid points per dimension (default 21).

        Returns
        -------
        axes : list of ndarray
            The 1D coordinate arrays, one per dimension.
        gap : ndarray
            The localizer gap on the grid, shape ``(ngrid_1, ..., ngrid_d)``.
        """
        if bounds is None:
            bounds = self._default_bounds()
        if np.isscalar(ngrid):
            ngrid = [int(ngrid)] * self.ndim
        axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, ngrid)]
        mesh = np.meshgrid(*axes, indexing="ij")
        gap = np.empty(mesh[0].shape, dtype=float)
        for idx in np.ndindex(*gap.shape):
            point = np.array([m[idx] for m in mesh])
            gap[idx] = self.localizer_gap(point, energy)
        return axes, gap

    def find_centers(
        self, energy=0.0, bounds=None, ngrid=21, gap_threshold=None, refine=True
    ):
        r"""
        Locate electron / Wannier centers as minima of the localizer gap.

        The localizer gap closes at the positions of the localized electronic
        states.  This routine scans the gap on a grid, identifies local minima, and
        (optionally) refines them with a local optimizer.

        Parameters
        ----------
        energy : float, optional
            The reference energy (default 0), inside the spectral gap of ``H``.
        bounds : sequence of (float, float), optional
            Per-dimension search ranges; defaults to the position-operator spans.
        ngrid : int or sequence of int, optional
            Grid resolution per dimension (default 21).
        gap_threshold : float, optional
            Only grid minima whose gap is below this value are kept.  If ``None``,
            a value of ``0.1 * kappa * (grid spacing)`` is used as a heuristic.
        refine : bool, optional
            If ``True`` (default) and SciPy is available, each candidate is refined
            with a derivative-free local minimization of the gap.

        Returns
        -------
        centers : ndarray
            The located centers, shape ``(M, ndim)``.
        gaps : ndarray
            The localizer gap at each center, shape ``(M,)``.
        """
        axes, gap = self.gap_scan(energy=energy, bounds=bounds, ngrid=ngrid)

        # Heuristic threshold based on the grid spacing.
        if gap_threshold is None:
            spacing = min(
                (ax[1] - ax[0]) if len(ax) > 1 else 1.0 for ax in axes
            )
            gap_threshold = 0.5 * self.kappa * spacing

        # Find local minima of the gap grid.
        candidates = []
        for idx in np.ndindex(*gap.shape):
            val = gap[idx]
            if val > gap_threshold:
                continue
            is_min = True
            for neighbor in _neighbors(idx, gap.shape):
                if gap[neighbor] < val:
                    is_min = False
                    break
            if is_min:
                point = np.array([axes[k][idx[k]] for k in range(self.ndim)])
                candidates.append(point)

        if refine and candidates:
            candidates = [
                self._refine_center(p, energy, axes) for p in candidates
            ]

        if not candidates:
            return np.empty((0, self.ndim)), np.empty((0,))

        # Merge duplicates that collapsed onto the same refined point.
        centers = _merge_points(np.array(candidates))
        gaps = np.array([self.localizer_gap(c, energy) for c in centers])
        order = np.lexsort(centers.T[::-1])
        return centers[order], gaps[order]

    def _refine_center(self, point, energy, axes):
        """Refine a candidate center by locally minimizing the localizer gap."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            return point
        spacing = np.array(
            [(ax[1] - ax[0]) if len(ax) > 1 else 1.0 for ax in axes]
        )
        res = minimize(
            lambda p: self.localizer_gap(p, energy),
            point,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10},
        )
        # Reject runaway refinements that leave the local cell.
        if np.all(np.abs(res.x - point) <= spacing):
            return res.x
        return point

    # ------------------------------------------------------------------ #
    # Localized states
    # ------------------------------------------------------------------ #
    def localized_state(self, point, energy=0.0):
        r"""
        Return the approximate localized electronic state at ``point``.

        The eigenvector of the localizer with the smallest-magnitude eigenvalue,
        projected onto the physical (first :math:`N`) components, gives the
        maximally localized state associated with the probe point.

        Parameters
        ----------
        point : array-like
            The probe point (typically a located center), length ``ndim``.
        energy : float, optional
            The reference energy (default 0).

        Returns
        -------
        ndarray
            A normalized length-``N`` complex vector in the one-particle basis.
        """
        evals, evecs = np.linalg.eigh(self.localizer(point, energy))
        imin = int(np.argmin(np.abs(evals)))
        s = len(self._gammas[0])
        # The 2N (or 4N) eigenvector splits into s blocks of length N.
        vec = evecs[:, imin].reshape(self.nbasis, s)[:, 0]
        nrm = np.linalg.norm(vec)
        if nrm == 0:
            # Fall back to the dominant Clifford component.
            blocks = evecs[:, imin].reshape(self.nbasis, s)
            vec = blocks[:, int(np.argmax(np.linalg.norm(blocks, axis=0)))]
            nrm = np.linalg.norm(vec)
        return vec / nrm

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_lattice(cls, hamiltonian, site_coords, orbitals_per_site=1, kappa=None):
        r"""
        Build a localizer for a tight-binding lattice from site coordinates.

        Parameters
        ----------
        hamiltonian : ndarray
            The single-particle (Bloch-free, real-space) Hamiltonian, Hermitian and
            of shape ``(N, N)`` with ``N = nsites * orbitals_per_site``.
        site_coords : ndarray
            Site coordinates, shape ``(nsites, d)`` (or ``(nsites,)`` for ``d=1``).
        orbitals_per_site : int, optional
            Number of orbitals/bands per site (default 1).  Orbitals on the same
            site share its coordinate.  The basis ordering is assumed to be
            site-major: ``[(site0,orb0), (site0,orb1), ..., (site1,orb0), ...]``.
        kappa : float, optional
            The scaling constant; see :class:`SpectralLocalizer`.
        """
        site_coords = np.asarray(site_coords, dtype=float)
        if site_coords.ndim == 1:
            site_coords = site_coords[:, None]
        nsites, d = site_coords.shape
        coords = np.repeat(site_coords, orbitals_per_site, axis=0)
        positions = [np.diag(coords[:, j]) for j in range(d)]
        return cls(hamiltonian, positions, kappa=kappa)

    @classmethod
    def from_model_system(cls, model_system, site_coords=None, kappa=None):
        r"""
        Build a localizer from a forte2 :class:`~forte2.system.ModelSystem`.

        The single-particle Hamiltonian is taken from ``model_system.ints_hcore()``.
        For a :class:`~forte2.system.HubbardModel` the site coordinates are inferred
        from ``nsites`` if not supplied.

        Parameters
        ----------
        model_system : ModelSystem
            The model system providing the core Hamiltonian.
        site_coords : ndarray, optional
            Site coordinates, shape ``(nsites, d)``.  Required unless the model can
            supply them (e.g. ``HubbardModel``).
        kappa : float, optional
            The scaling constant; see :class:`SpectralLocalizer`.
        """
        H = np.asarray(model_system.ints_hcore())
        if site_coords is None:
            site_coords = _hubbard_site_coords(model_system)
        if site_coords is None:
            raise ValueError(
                "`site_coords` must be provided for this model system."
            )
        return cls.from_lattice(H, site_coords, kappa=kappa)

    @classmethod
    def from_scf(cls, scf, spin=0, energy=None, kappa=None):
        r"""
        Build a localizer for a molecular mean-field (SCF) solution.

        The Hamiltonian is the Fock operator in the (orthonormal) MO basis, i.e. the
        diagonal matrix of orbital energies, and the position operators are the
        Cartesian dipole integrals transformed to the MO basis.  The default
        reference energy is the middle of the HOMO-LUMO gap, which is the natural
        reference for the local topological index of the occupied manifold.

        Note that the localizer gap of a finite molecule closes at probe points
        ``(lambda, E)`` where the Fock operator has an orbital of energy ``~E``
        localized near ``lambda``; locating individual electrons therefore requires
        scanning the reference energy as well as the probe position.

        Parameters
        ----------
        scf : SCFBase
            A converged SCF object (e.g. :class:`~forte2.scf.RHF`).
        spin : int, optional
            Which spin block of the MO coefficients/energies to use for unrestricted
            references (default 0).
        energy : float, optional
            The reference energy.  Defaults to the midpoint of the HOMO-LUMO gap.
        kappa : float, optional
            The scaling constant; see :class:`SpectralLocalizer`.

        Returns
        -------
        SpectralLocalizer
            A 3D localizer in the molecular MO basis.
        energy : float
            The reference energy used is stored on the returned object as
            ``reference_energy``.
        """
        from forte2 import integrals

        system = scf.system
        C = _select_spin(scf.C, spin)
        eps = np.asarray(_select_spin(scf.eps, spin), dtype=float)

        # Position integrals in the AO basis: [overlap, x, y, z].
        ao = integrals.emultipole1(system)
        mo_pos = [C.conj().T @ np.asarray(ao[a]) @ C for a in (1, 2, 3)]

        H = np.diag(eps)

        if energy is None:
            energy = _homo_lumo_midpoint(scf, eps, spin)

        obj = cls(H, mo_pos, kappa=kappa)
        obj.reference_energy = float(energy)
        return obj


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #
def _neighbors(idx, shape):
    """Yield the orthogonal grid neighbors of a multi-index."""
    for axis in range(len(shape)):
        for step in (-1, 1):
            nb = list(idx)
            nb[axis] += step
            if 0 <= nb[axis] < shape[axis]:
                yield tuple(nb)


def _merge_points(points, tol=1e-4):
    """Collapse points that are within ``tol`` of one another."""
    merged = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= tol for q in merged):
            merged.append(p)
    return np.array(merged)


def _hubbard_site_coords(model_system):
    """Infer site coordinates from a HubbardModel, else return None."""
    nsites = getattr(model_system, "nsites", None)
    if nsites is None:
        return None
    if isinstance(nsites, int):
        return np.arange(nsites, dtype=float)[:, None]
    if isinstance(nsites, tuple) and len(nsites) == 2:
        nx, ny = nsites
        # Site index convention in HubbardModel is i*ny + j (see system.py).
        coords = np.array([[i, j] for i in range(nx) for j in range(ny)], dtype=float)
        return coords
    return None


def _select_spin(value, spin):
    """Return the ``spin``-th entry of a per-spin list, or ``value`` itself."""
    if isinstance(value, (list, tuple)):
        return value[spin if spin < len(value) else 0]
    return value


def _homo_lumo_midpoint(scf, eps, spin):
    """Reference energy at the middle of the HOMO-LUMO gap."""
    nel = getattr(scf, "nel", None)
    # Determine the number of occupied orbitals in this spin channel.
    if isinstance(scf.C, (list, tuple)) and len(scf.C) == 2:
        nocc_list = [getattr(scf, "na", None), getattr(scf, "nb", None)]
        nocc = nocc_list[spin] if nocc_list[spin] is not None else None
    else:
        nocc = nel // 2 if nel is not None else None
    if nocc is None or nocc <= 0 or nocc >= len(eps):
        # Fall back to 0 (a safe default if occupation is unknown).
        return 0.0
    return 0.5 * (eps[nocc - 1] + eps[nocc])
