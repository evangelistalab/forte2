import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from forte2.base_classes import Method
from forte2.base_classes.rebuild import (
    list_method_chain,
    rebind_method_chain,
    rebuild_method_chain,
    project_scf_guess,
)
from forte2.helpers import logger
from forte2.orbitals import ci_overlap_matrix
from .fd_gradient_helper import central_stencil, finite_difference

# Multiples of the linear estimate, and an absolute floor in Eh, beyond which a
# displaced energy is treated as having landed on a different solution branch.
_ENERGY_JUMP_FACTOR = 50.0
_ENERGY_JUMP_FLOOR = 1.0e-3
# A displaced root (or degenerate manifold) whose overlap with its reference
# counterpart has a singular value below this changed character over the step.
_MIN_OVERLAP_SINGULAR_VALUE = 0.9
_OVERLAP_ALGORITHMS = ("biorthogonal", "naive")


@dataclass
class FiniteDifference(Method):
    r"""
    Nuclear gradients and nonadiabatic couplings of a method, by central
    differences.

    Attach it to any method that can be rebuilt at a new geometry, and it
    provides the same ``gradient()`` contract as the analytic implementations::

        fd = FiniteDifference(step=1.0e-3)(mc)
        g = fd.gradient(root=1)

    Because it exposes ``gradient()``, it can drive a geometry optimization of a
    method that has no analytic gradient::

        GeometryOptimizer(g_tol=1.0e-5)(FiniteDifference()(mc)).run()

    For a CI or MCOptimizer with more than one root, it also provides
    ``nonadiabatic_coupling()``, from the overlaps between the reference and
    the displaced wavefunctions::

        fd = FiniteDifference(compute_nac=True)(mc)
        d = fd.nonadiabatic_coupling(ket=1, bra=0)
        g_0, g_1 = fd.gradient(root=0), fd.gradient(root=1)

    Every displacement rebuilds the whole upstream chain at the displaced
    geometry and reruns it, so a sweep of displacements costs ``npoints``
    evaluations of the upstream method per differentiated component, or
    ``npoints * 3 * natoms`` by default. One sweep gives the gradients of every
    root and, if it collects the overlaps, every coupling.

    Parameters
    ----------
    step : float, optional, default=1.0e-3
        Cartesian displacement in Bohr.
    npoints : int, optional, default=4
        Central-difference stencil size; one of 2, 4, or 6.
    components : list[tuple[int, int]], optional
        The ``(atom, xyz)`` Cartesian components to differentiate. All of them
        if None. The results are NaN for the others.
    project_orbitals : bool, optional, default=True
        Whether to seed each displaced calculation with orbitals projected from
        the reference geometry.
    energy_accessor : callable, optional
        Custom extractor for the energy that ``gradient()`` differentiates,
        overriding ``method.E``. Can't be combined with ``gradient(root=...)``.
    residual_tol : float, optional, default=1.0e-6
        Warn when the residual net force or net torque of a gradient exceeds
        this value, in Eh/Bohr.
    compute_nac : bool, optional, default=False
        Whether every sweep also collects the overlaps that
        ``nonadiabatic_coupling()`` needs. The upstream method must be a CI or
        MCOptimizer with a CISolver or RelCISolver and more than one root,
        which is checked when this is attached. If False, calling
        ``nonadiabatic_coupling()`` after ``gradient()`` takes a second sweep.
    degeneracy_tol : float, optional, default=1.0e-6
        Roots closer in energy than this, in Eh, are treated as one degenerate
        manifold, such as a Kramers pair.
    overlap_algorithm : {"biorthogonal", "naive"}, optional, default="biorthogonal"
        The algorithm for :func:`forte2.orbitals.ci_overlap_matrix`. GAS
        wavefunctions need ``"naive"``.

    Attributes
    ----------
    E : float | None
        The reference-geometry energy that ``gradient()`` differentiates, or
        None if the method reports one energy per root.
    E_ci : NDArray | None
        The reference-geometry root energies that ``gradient(root=...)``
        differentiates, or None if the method has none.
    net_force : NDArray | None
        Sum of the rows of the last gradient, shape ``(3,)``, if it was
        differentiated in every component. Exactly zero for an exact gradient,
        so its magnitude bounds the numerical error.
    net_torque : NDArray | None
        Sum of ``r_A x g_A`` of the last gradient, shape ``(3,)``. Also exactly
        zero for an exact gradient.
    anti_hermiticity_residual : float | None
        Largest ``||D + D^H||`` over the differentiated components, where ``D``
        is the matrix of couplings between all roots. Exactly zero for exact
        couplings.
    min_overlap_singular_value : float | None
        Smallest singular value, over all displacements, of the overlap between
        a reference root or degenerate manifold and its displaced counterpart.
        It stays close to 1 unless a displaced root changed character.
    n_evaluations : int
        How many times the upstream method was run at a displaced geometry.
    wall_time : float
        Time spent in sweeps of displacements, in seconds.
    """

    step: float = 1.0e-3
    npoints: int = 4
    components: list[tuple[int, int]] | None = None
    project_orbitals: bool = True
    energy_accessor: Callable | None = None
    residual_tol: float = 1.0e-6
    compute_nac: bool = False
    degeneracy_tol: float = 1.0e-6
    overlap_algorithm: str = "biorthogonal"

    E: float | None = field(default=None, init=False)
    E_ci: NDArray | None = field(default=None, init=False)
    net_force: NDArray | None = field(default=None, init=False)
    net_torque: NDArray | None = field(default=None, init=False)
    anti_hermiticity_residual: float | None = field(default=None, init=False)
    min_overlap_singular_value: float | None = field(default=None, init=False)
    n_evaluations: int = field(default=0, init=False)
    wall_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.requires = {"system", "mos"}
        self.provides = {"system", "mos"}
        if not np.isscalar(self.step) or self.step <= 0.0:
            raise ValueError(f"step must be a positive number, but got {self.step}.")
        central_stencil(self.npoints)  # validates npoints
        self._components = _validate_components(self.components)
        if not np.isscalar(self.residual_tol) or self.residual_tol <= 0.0:
            raise ValueError(
                f"residual_tol must be a positive number, but got {self.residual_tol}."
            )
        if self.energy_accessor is not None and not callable(self.energy_accessor):
            raise ValueError(
                f"energy_accessor must be callable, but got {self.energy_accessor}."
            )
        if not np.isscalar(self.degeneracy_tol) or self.degeneracy_tol < 0.0:
            raise ValueError(
                "degeneracy_tol must be a nonnegative number, but got "
                f"{self.degeneracy_tol}."
            )
        if self.overlap_algorithm not in _OVERLAP_ALGORITHMS:
            raise ValueError(
                f"overlap_algorithm must be one of {_OVERLAP_ALGORITHMS}, but got "
                f"{self.overlap_algorithm!r}."
            )
        self._scratch_chain = None
        self._clear()

    def __call__(self, method):
        """Attach to the upstream method to differentiate."""
        self._register_parent_method(method)
        if self.compute_nac:
            _check_nac_parent(method)
        return self

    def reset(self):
        """
        Invalidate the cached derivatives so a rebind to a new reference
        geometry recomputes them, instead of returning the previous geometry's.
        self._scratch_chain is kept: the next sweep rebinds it rather than
        rebuilding it.
        """
        self._clear()
        return super().reset()

    def _clear(self):
        self.E = None
        self.E_ci = None
        self.net_force = None
        self.net_torque = None
        self.anti_hermiticity_residual = None
        self.min_overlap_singular_value = None
        self.n_evaluations = 0
        self.wall_time = 0.0
        self._reference_energies = None
        self._energy_derivatives = None
        self._nac = None
        self._manifolds = None
        self._manifold_of = None

    def run(self):
        """
        Run the upstream method at the reference geometry.

        The displacements are deferred to :meth:`gradient` and
        :meth:`nonadiabatic_coupling`, so attaching this to a method costs
        nothing extra when only the energy is wanted.

        Returns
        -------
        FiniteDifference
            The executed object.
        """
        if not self.parent_method.executed:
            self.parent_method.run()

        self.system = self.parent_method.system
        self.mos = self.parent_method.mos
        self.E, self.E_ci = self._get_energies(self.parent_method)
        self._reference_energies = self._pack_energies(self.E, self.E_ci)
        self.executed = True
        return self

    def gradient(self, root: int | None = None) -> NDArray:
        """
        Compute a nuclear gradient by central differences.

        Parameters
        ----------
        root : int, optional
            The absolute root whose energy to differentiate, from ``E_ci``. If
            None, the gradient is that of ``E``.

        Returns
        -------
        NDArray
            Gradient with shape ``(natoms, 3)`` in Hartree/Bohr.

        Raises
        ------
        ValueError
            If ``root`` is out of range, if it's given with ``energy_accessor``
            or for a method without root energies, or if it's omitted for a
            method that only reports one energy per root.
        """
        if not self.executed:
            self.run()
        column = self._energy_column(root)
        if self._energy_derivatives is None:
            self._sweep(with_overlaps=self.compute_nac)

        gradient = self._energy_derivatives[..., column].copy()
        self.net_force = self.net_torque = None
        if self._components is None:
            coordinates = np.asarray(self.system.atomic_positions, dtype=float)
            self.net_force = gradient.sum(axis=0)
            self.net_torque = np.cross(coordinates, gradient).sum(axis=0)
        self._print_gradient(gradient, root)
        self._warn_on_residuals()
        return gradient

    def nonadiabatic_coupling(
        self, ket: int, bra: int, *, energy_gap_weighted: bool = False
    ) -> NDArray:
        r"""
        Compute the nonadiabatic coupling
        :math:`\langle\mathrm{bra}|\nabla_R\,\mathrm{ket}\rangle` by central
        differences.

        The coupling is the derivative of the overlap between the reference bra
        and the displaced ket. Before differencing, every displaced root is
        aligned with its reference root: its phase is chosen so that the two
        overlap positively, and a degenerate manifold, such as a Kramers pair,
        is rotated as a whole to match its reference counterpart.

        The basis functions move with the atoms, so the coupling includes their
        contribution (the CSF term) and isn't translationally invariant. It
        carries the arbitrary phases of the reference bra and ket, the same for
        every component, so only its magnitude is unique.

        Parameters
        ----------
        ket, bra : int
            Absolute root indices in the upstream CI solver.
        energy_gap_weighted : bool, optional, default=False
            Return the coupling times :math:`E_\mathrm{ket}-E_\mathrm{bra}`.

        Returns
        -------
        NDArray
            Coupling with shape ``(natoms, 3)`` in inverse Bohr, or in
            Hartree/Bohr if ``energy_gap_weighted`` is True. Complex if the
            upstream method is two-component.

        Raises
        ------
        TypeError
            If the upstream method isn't a CI or MCOptimizer with a CISolver
            or RelCISolver.
        ValueError
            If the upstream method has one root, if ``ket`` or ``bra`` is out
            of range, or if they're the same root or degenerate roots.
        """
        _check_nac_parent(self.parent_method)
        nroots = self.parent_method.ci_solver.sa_info.nroots_sum
        for name, root in (("ket", ket), ("bra", bra)):
            if not _is_index(root, nroots):
                raise ValueError(f"{name} must be in [0, {nroots}), but got {root}.")
        if ket == bra:
            raise ValueError(
                f"ket and bra must be different roots, but both are {ket}."
            )
        if not self.executed:
            self.run()
        self._prepare_couplings()
        if self._manifold_of[ket] == self._manifold_of[bra]:
            raise ValueError(
                f"Roots {ket} and {bra} are degenerate within "
                f"degeneracy_tol={self.degeneracy_tol:.1e} Eh, so the coupling "
                "between them isn't defined."
            )
        if self._nac is None:
            if self._energy_derivatives is not None:
                logger.log_info1(
                    "compute_nac=False: running a second finite-difference sweep "
                    "for the couplings."
                )
            self._sweep(with_overlaps=True)

        coupling = self._nac[..., bra, ket]
        coupling = (coupling if self.two_component else coupling.real).copy()
        if energy_gap_weighted:
            energies = np.asarray(self.parent_method.E_ci).real
            coupling *= energies[ket] - energies[bra]
        return coupling

    def _get_energies(self, method):
        """
        Extract the energy that gradient() differentiates, and the root
        energies that gradient(root=...) does.
        """
        if self.energy_accessor is not None:
            energy = self.energy_accessor(method)
            if np.ndim(energy) != 0:
                raise ValueError(
                    f"energy_accessor on {type(method).__name__} returned a "
                    "non-scalar energy."
                )
            return _real(energy), None

        energies = np.asarray(method.E)
        energy = _real(energies.reshape(-1)[0]) if energies.size == 1 else None
        if hasattr(method, "E_ci"):
            roots = np.asarray(method.E_ci)
        elif energies.ndim > 0:
            roots = energies
        else:
            return energy, None
        return energy, np.array([_real(e) for e in roots.reshape(-1)])

    def _pack_energies(self, energy, root_energies):
        """The differentiated energies as one vector, NaN for a missing E."""
        energy = np.nan if energy is None else energy
        return np.array([energy, *([] if root_energies is None else root_energies)])

    def _energy_column(self, root):
        """Validate `root` and return its column in the energy derivatives."""
        name = type(self.parent_method).__name__
        if root is None:
            if self.E is None:
                raise ValueError(
                    f"{name} reports {len(self.E_ci)} energies; pass root to choose "
                    "which one to differentiate."
                )
            return 0
        if self.energy_accessor is not None:
            raise ValueError("root can't be combined with energy_accessor.")
        if self.E_ci is None:
            raise ValueError(f"{name} reports a single energy, so root must be None.")
        if not _is_index(root, len(self.E_ci)):
            raise ValueError(f"root must be in [0, {len(self.E_ci)}), but got {root}.")
        return 1 + root

    def _prepare_couplings(self):
        """
        Group the roots into degenerate manifolds, and check that the overlaps
        apply to the reference wavefunction, before any displacement.
        """
        if self._manifolds is not None:
            return
        _check_nac_parent(self.parent_method)
        S = ci_overlap_matrix(
            self.parent_method, self.parent_method, algorithm=self.overlap_algorithm
        )
        deviation = np.abs(S - np.eye(S.shape[0])).max()
        if deviation > 1.0e-6:
            logger.log_warning(
                "The reference roots aren't orthonormal (largest deviation "
                f"{deviation:.2e}), so the couplings are unreliable: converge "
                "the CI solver more tightly."
            )
        energies = np.asarray(self.parent_method.E_ci).real
        self._manifolds = _degenerate_manifolds(
            energies, _overlap_blocks(self.parent_method), self.degeneracy_tol
        )
        self._manifold_of = {
            root: i for i, manifold in enumerate(self._manifolds) for root in manifold
        }

    def _sweep(self, with_overlaps):
        """
        Differentiate the energies, and the aligned overlaps with the reference
        roots if `with_overlaps`, over one sweep of displacements.
        """
        coordinates = np.asarray(self.system.atomic_positions, dtype=float)
        if self._components is not None:
            natoms = coordinates.shape[0]
            for atom, _ in self._components:
                if atom >= natoms:
                    raise ValueError(
                        f"components refers to atom {atom}, but there are only "
                        f"{natoms} atoms."
                    )
        if with_overlaps:
            self._prepare_couplings()
        self._warn_if_upstream_unconverged()
        self._print_start(coordinates, with_overlaps)

        displaced_energies = []
        singular_values = []

        def evaluate(displaced):
            method = self._run_at(displaced)
            energies = self._pack_energies(*self._get_energies(method))
            displaced_energies.append(energies)
            if not with_overlaps:
                return energies
            S = ci_overlap_matrix(
                self.parent_method, method, algorithm=self.overlap_algorithm
            )
            S, smin = _align_to_reference(S, self._manifolds)
            singular_values.append(smin)
            return np.concatenate([energies, S.reshape(-1)])

        # The upstream method runs once per displacement; let it report only if
        # the caller asked for more than the default detail.
        verbosity = logger.get_verbosity_level()
        logger.set_verbosity_level(max(verbosity - 1, 0))
        start = time.monotonic()
        try:
            derivative = finite_difference(
                evaluate,
                coordinates,
                step=self.step,
                npoints=self.npoints,
                components=self._components,
                progress=self._report_progress,
            )
        finally:
            logger.set_verbosity_level(verbosity)
        self.wall_time += time.monotonic() - start

        derivative = self._scatter(derivative, coordinates.shape)
        nenergies = len(self._reference_energies)
        self._energy_derivatives = derivative[..., :nenergies].real.copy()
        self._warn_on_energy_jumps(np.array(displaced_energies))
        if with_overlaps:
            nroots = len(self.parent_method.E_ci)
            D = derivative[..., nenergies:].reshape(coordinates.shape + (nroots,) * 2)
            self._nac = D
            self.min_overlap_singular_value = float(min(singular_values))
            self.anti_hermiticity_residual = max(
                float(np.linalg.norm(D[c] + D[c].conj().T))
                for c in self._differentiated(coordinates.shape)
            )
            self._print_couplings()
            self._warn_on_overlaps()
        logger.log_info1(f"Wall time: {self.wall_time:.2f} s")

    def _run_at(self, coordinates):
        """
        Evaluate the upstream chain at `coordinates` and return its last stage.

        Reuses a single scratch copy of the upstream chain across every
        displacement (rebound in place, not rebuilt), rather than allocating a
        fresh chain per stencil point. `self.parent_method` itself is never
        touched, so it stays the untouched reference-geometry seed source for
        every displacement, in any order.
        """
        system = self.system.with_geometry(coordinates)
        if self._scratch_chain is None:
            self._scratch_chain = rebuild_method_chain(self.parent_method, system)
        else:
            rebind_method_chain(self._scratch_chain, system)
        if self.project_orbitals:
            # Always from the reference geometry, never from the previous
            # displacement: see the class docstring.
            project_scf_guess(self.parent_method, self._scratch_chain)
        self._scratch_chain.run()
        self.n_evaluations += 1
        return self._scratch_chain

    def _differentiated(self, shape):
        """The (atom, xyz) components of this calculation."""
        return list(np.ndindex(shape)) if self._components is None else self._components

    def _scatter(self, derivative, shape):
        """Expand a derivative over `components` to all components, NaN elsewhere."""
        if self._components is None:
            return derivative
        full = np.full(shape + derivative.shape[1:], np.nan, dtype=derivative.dtype)
        for i, component in enumerate(self._components):
            full[component] = derivative[i]
        return full

    def _warn_if_upstream_unconverged(self):
        """Warn if any stage of the reference chain stopped short of convergence."""
        for stage in list_method_chain(self.parent_method):
            if getattr(stage, "converged", True) is False:
                logger.log_warning(
                    f"{type(stage).__name__} did not converge at the reference "
                    "geometry; the finite differences are unreliable."
                )

    def _warn_on_residuals(self):
        """Warn when the measured invariance residuals of a gradient are large."""
        for name, residual in (
            ("net force", self.net_force),
            ("net torque", self.net_torque),
        ):
            if residual is None:
                continue
            norm = float(np.linalg.norm(residual))
            if norm > self.residual_tol:
                logger.log_warning(
                    f"Finite-difference residual {name} is {norm:.3e} Eh/Bohr, above "
                    f"residual_tol={self.residual_tol:.3e}. The gradient is likely "
                    "inaccurate at that level: tighten the convergence thresholds of "
                    f"{type(self.parent_method).__name__} or increase step."
                )

    def _warn_on_energy_jumps(self, displaced_energies):
        """
        Warn when a displaced energy is far from the reference energy.

        Over a displacement this small an energy should change by roughly
        ``offset * step * |gradient|``. A much larger change means that
        displacement converged to something else -- a different SCF solution, or
        a different CI root -- in which case the difference quotient straddles a
        discontinuity and the derivatives are meaningless rather than merely
        noisy.
        """
        finite = np.isfinite(self._reference_energies)
        offsets, _, _ = central_stencil(self.npoints)
        slopes = np.abs(self._energy_derivatives[..., finite]).reshape(-1, finite.sum())
        expected = max(abs(o) for o in offsets) * self.step * np.nanmax(slopes, axis=0)
        observed = np.abs(
            displaced_energies[:, finite] - self._reference_energies[finite]
        ).max(axis=0)
        # A generous factor: this should only fire on a qualitative change, not
        # on the second-order curvature the linear estimate ignores.
        threshold = np.maximum(_ENERGY_JUMP_FACTOR * expected, _ENERGY_JUMP_FLOOR)
        if np.any(observed > threshold):
            worst = int(np.argmax(observed / threshold))
            logger.log_warning(
                f"A displaced energy differs from the reference by "
                f"{observed[worst]:.3e} Eh, far more than the {expected[worst]:.3e} "
                "Eh expected from its gradient. Some displacement probably "
                "converged to a different SCF solution or CI root, which "
                "invalidates the finite differences."
            )

    def _warn_on_overlaps(self):
        """Warn when a displaced root lost its overlap with its reference root."""
        if self.min_overlap_singular_value < _MIN_OVERLAP_SINGULAR_VALUE:
            logger.log_warning(
                "A displaced root overlaps its reference root with a singular "
                f"value of only {self.min_overlap_singular_value:.3f}, so it "
                "changed character or order over the step, which invalidates the "
                "couplings. Reduce step, or adjust degeneracy_tol if nearly "
                "degenerate roots were treated separately."
            )

    def _report_progress(self, done, total):
        logger.log_info2(f"  finite-difference displacement {done}/{total}")

    def _print_start(self, coordinates, with_overlaps):
        ncomponents = len(self._differentiated(coordinates.shape))
        logger.log_info1("\n==> FINITE-DIFFERENCE DERIVATIVES <==")
        logger.log_info1(f"Method: {type(self.parent_method).__name__}")
        logger.log_info1(f"Atoms: {len(coordinates)}")
        logger.log_info1(f"Components: {ncomponents}")
        logger.log_info1(f"Stencil: {self.npoints}-point central")
        logger.log_info1(f"Step: {self.step:.3e} Bohr")
        logger.log_info1(f"Evaluations: {self.npoints * ncomponents}")
        logger.log_info1(
            f"Reference-geometry orbital projection: "
            f"{'on' if self.project_orbitals else 'off'}"
        )
        logger.log_info1(
            f"Nonadiabatic couplings: "
            f"{self.overlap_algorithm + ' overlaps' if with_overlaps else 'off'}"
        )

    def _print_gradient(self, gradient, root):
        label = "" if root is None else f" of root {root}"
        logger.log_info1(f"\nFinite-difference gradient{label} [Eh/Bohr]:")
        logger.log_info1("-" * 52)
        logger.log_info1(f"{'Atom':>5} {'X':>15} {'Y':>15} {'Z':>15}")
        logger.log_info1("-" * 52)
        for atom, row in enumerate(gradient):
            logger.log_info1(f"{atom:>5} {row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}")
        logger.log_info1("-" * 52)
        if self.net_force is not None:
            logger.log_info1(f"Norm: {np.linalg.norm(gradient):12.6e} Eh/Bohr")
            # Both vanish for an exact gradient, so they measure the numerical error.
            logger.log_info1(
                f"Residual net force:  {np.linalg.norm(self.net_force):12.6e}"
            )
            logger.log_info1(
                f"Residual net torque: {np.linalg.norm(self.net_torque):12.6e}"
            )

    def _print_couplings(self):
        manifolds = ", ".join(
            "{" + ", ".join(str(r) for r in manifold) + "}"
            for manifold in self._manifolds
        )
        logger.log_info1(f"Root manifolds: {manifolds}")
        logger.log_info1(
            "Smallest overlap singular value: " f"{self.min_overlap_singular_value:.8f}"
        )
        # Vanishes for exact couplings, so it measures the numerical error.
        logger.log_info1(
            f"Residual ||D + D^H||: {self.anti_hermiticity_residual:12.6e} 1/Bohr"
        )


def _validate_components(components):
    """Return `components` as a list of (atom, xyz) tuples, or None."""
    if components is None:
        return None
    validated = []
    for component in components:
        if (
            len(component) != 2
            or not all(_is_index(i, np.inf) for i in component)
            or component[1] > 2
        ):
            raise ValueError(
                "components must hold (atom, xyz) pairs of nonnegative integers "
                f"with xyz in 0, 1, 2, but got {component}."
            )
        validated.append((int(component[0]), int(component[1])))
    if not validated:
        raise ValueError("components must not be empty.")
    return validated


def _is_index(value, size):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, np.integer))
        and 0 <= value < size
    )


def _real(energy):
    """Return `energy` as a float, rejecting a significant imaginary part."""
    if np.iscomplexobj(energy):
        if abs(energy.imag) > 1.0e-10:
            raise ValueError(
                f"Energy {energy} has a significant imaginary part and cannot "
                "be differentiated."
            )
        energy = energy.real
    return float(energy)


def _check_nac_parent(method):
    """Raise unless `method` can provide nonadiabatic couplings."""
    from forte2.base_classes import ActiveSpaceDriver
    from forte2.ci import CISolver, RelCISolver

    name = type(method).__name__
    if not isinstance(method, ActiveSpaceDriver):
        raise TypeError(
            f"Nonadiabatic couplings require a CI or MCOptimizer, but got {name}."
        )
    if not isinstance(method.ci_solver, (CISolver, RelCISolver)):
        raise TypeError(
            "Nonadiabatic couplings require a CISolver or RelCISolver, but got "
            f"{type(method.ci_solver).__name__}."
        )
    nroots = method.ci_solver.sa_info.nroots_sum
    if nroots < 2:
        raise ValueError(
            f"Nonadiabatic couplings require more than one root, but {name} has "
            f"{nroots}."
        )


def _overlap_blocks(method):
    """
    Label each root by its numbers of alpha and beta electrons. Roots with
    different labels never overlap; two-component roots all share one label.
    """
    ci_solver = method.ci_solver
    labels = []
    for root in range(ci_solver.sa_info.nroots_sum):
        state = ci_solver.sa_info.states[ci_solver._get_state_root(root)[0]]
        labels.append(None if method.two_component else (state.na, state.nb))
    return labels


def _degenerate_manifolds(energies, labels, tol):
    """
    Group roots with the same label into manifolds whose energies are
    connected by gaps smaller than `tol`.
    """
    manifolds = []
    for label in dict.fromkeys(labels):
        roots = sorted(
            (r for r in range(len(energies)) if labels[r] == label),
            key=lambda r: energies[r],
        )
        manifold = [roots[0]]
        for previous, root in zip(roots[:-1], roots[1:]):
            if energies[root] - energies[previous] < tol:
                manifold.append(root)
            else:
                manifolds.append(sorted(manifold))
                manifold = [root]
        manifolds.append(sorted(manifold))
    return manifolds


def _align_to_reference(S, manifolds):
    """
    Rotate the displaced roots, the columns of the overlap `S` with the
    reference roots, within each manifold so that its diagonal block of `S`
    becomes Hermitian positive definite. For a single root, this chooses its
    phase so that it overlaps its reference root positively.

    Returns the aligned overlap and the smallest singular value of the
    diagonal blocks.
    """
    S = S.copy()
    smin = np.inf
    for manifold in manifolds:
        W, s, Vh = np.linalg.svd(S[np.ix_(manifold, manifold)])
        S[:, manifold] = S[:, manifold] @ (Vh.conj().T @ W.conj().T)
        smin = min(smin, s.min())
    return S, smin
