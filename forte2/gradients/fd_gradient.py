"""Nuclear gradients by finite differences of the energy."""

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from forte2.base_classes import Method
from forte2.base_classes.rebuild import (
    list_method_chain,
    rebind_method_chain,
    rebuild_method_chain,
    seed_chain_orbitals,
)
from forte2.helpers import logger
from .fd_gradient_helper import central_stencil, finite_difference

# Multiples of the linear estimate, and an absolute floor in Eh, beyond which a
# displaced energy is treated as having landed on a different solution branch.
_ENERGY_JUMP_FACTOR = 50.0
_ENERGY_JUMP_FLOOR = 1.0e-3


@dataclass
class FDGradient(Method):
    r"""
    Nuclear gradient of a method, by central differences of its energy.

    Attach it to any method that can be rebuilt at a new geometry, and it
    provides the same ``gradient()`` contract as the analytic implementations::

        fd = FDGradient(step=1.0e-3)(mc)
        fd.run().gradient()

    Because it exposes ``gradient()``, it can drive a geometry optimization of a
    method that has no analytic gradient::

        GeometryOptimizer(g_tol=1.0e-5)(FDGradient()(mc)).run()

    Every displacement rebuilds the whole upstream chain at the displaced
    geometry and reruns it, so the cost is ``npoints * 3 * natoms`` evaluations
    of the upstream method.

    Parameters
    ----------
    step : float, optional, default=1.0e-3
        Cartesian displacement in Bohr.
    npoints : int, optional, default=4
        Central-difference stencil size; one of 2, 4, or 6.
    root : int, optional
        Which root to differentiate, for upstream methods that report several
        energies. Required when the upstream method has more than one root.
    project_orbitals : bool, optional, default=True
        Whether to seed each displaced calculation with orbitals projected from
        the reference geometry.
    residual_tol : float, optional, default=1.0e-6
        Warn when the residual net force or net torque exceeds this value, in
        Eh/Bohr.

    Attributes
    ----------
    E : float
        The reference-geometry energy that was differentiated.
    net_force : NDArray
        Sum of the gradient rows, shape ``(3,)``. Exactly zero for an exact
        gradient, so its magnitude bounds the numerical error.
    net_torque : NDArray
        Sum of ``r_A x g_A``, shape ``(3,)``. Also exactly zero for an exact
        gradient.
    n_evaluations : int
        How many times the upstream method was run.
    """

    step: float = 1.0e-3
    npoints: int = 4
    root: int | None = None
    project_orbitals: bool = True
    residual_tol: float = 1.0e-6

    E: float | None = field(default=None, init=False)
    net_force: NDArray | None = field(default=None, init=False)
    net_torque: NDArray | None = field(default=None, init=False)
    n_evaluations: int = field(default=0, init=False)
    wall_time: float | None = field(default=None, init=False)

    def __post_init__(self):
        self.requires = {"system", "mos"}
        self.provides = {"system", "mos"}
        if not np.isscalar(self.step) or self.step <= 0.0:
            raise ValueError(f"step must be a positive number, but got {self.step}.")
        central_stencil(self.npoints)  # validates npoints
        if self.root is not None and self.root < 0:
            raise ValueError(f"root must be non-negative, but got {self.root}.")
        if not np.isscalar(self.residual_tol) or self.residual_tol <= 0.0:
            raise ValueError(
                f"residual_tol must be a positive number, but got {self.residual_tol}."
            )
        self._gradient = None
        self._displaced_energies = []
        self._scratch_chain = None

    def __call__(self, method):
        """Attach to the upstream method whose energy will be differentiated."""
        self._register_parent_method(method)
        return self

    def reset(self):
        """
        Invalidate the cached gradient/energy so a rebind to a new reference
        geometry recomputes them, instead of returning the previous geometry's
        cached gradient() result. self._scratch_chain is kept: it gets rebound
        to the new displacements in _energy_at rather than rebuilt.
        """
        self._gradient = None
        self._displaced_energies = []
        return super().reset()

    def run(self):
        """
        Run the upstream method at the reference geometry.

        The displacements are not made here; they are deferred to
        :meth:`gradient` so that attaching this to a method costs nothing extra
        when only the energy is wanted.

        Returns
        -------
        FDGradient
            The executed object.
        """
        if not self.parent_method.executed:
            self.parent_method.run()

        self.system = self.parent_method.system
        self.mos = self.parent_method.mos
        self.E = self._scalar_energy(self.parent_method)
        self.executed = True
        return self

    def gradient(self) -> NDArray:
        """
        Compute the nuclear gradient by central differences.

        Returns
        -------
        NDArray
            Gradient with shape ``(natoms, 3)`` in Hartree/Bohr.
        """
        if not self.executed:
            self.run()
        if self._gradient is None:
            self._gradient = self._compute_gradient()
        return self._gradient.copy()

    def _compute_gradient(self):
        coordinates = np.asarray(self.system.atomic_positions, dtype=float)
        self.n_evaluations = 0
        self._displaced_energies = []
        self._warn_if_upstream_unconverged()
        self._print_start(coordinates)

        # The upstream method runs once per displacement; let it report only if
        # the caller asked for more than the default detail.
        verbosity = logger.get_verbosity_level()
        logger.set_verbosity_level(min(verbosity - 1, 0))
        start = time.monotonic()
        try:
            gradient = finite_difference(
                self._energy_at,
                coordinates,
                step=self.step,
                npoints=self.npoints,
                progress=self._report_progress,
            )
        finally:
            logger.set_verbosity_level(verbosity)
        self.wall_time = time.monotonic() - start

        self.net_force = gradient.sum(axis=0)
        self.net_torque = np.cross(coordinates, gradient).sum(axis=0)
        self._print_finish(gradient)
        self._warn_on_residuals()
        self._warn_on_energy_jumps(gradient)
        return gradient

    def _energy_at(self, coordinates):
        """
        Evaluate the upstream chain at `coordinates` and return its energy.

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
            seed_chain_orbitals(self.parent_method, self._scratch_chain)
        self._scratch_chain.run()
        energy = self._scalar_energy(self._scratch_chain)
        self._displaced_energies.append(energy)
        return energy

    def _scalar_energy(self, method):
        """Extract the single energy to differentiate from `method`."""
        energies = np.asarray(method.E).reshape(-1)
        if self.root is None:
            if energies.size != 1:
                raise ValueError(
                    f"{type(method).__name__} reports {energies.size} energies; set "
                    "root to choose which one to differentiate."
                )
            value = energies[0]
        else:
            if self.root >= energies.size:
                raise ValueError(
                    f"root={self.root} is out of range: {type(method).__name__} "
                    f"reports {energies.size} energies."
                )
            value = energies[self.root]

        if np.iscomplexobj(value):
            if abs(value.imag) > 1.0e-10:
                raise ValueError(
                    f"Energy {value} has a significant imaginary part and cannot "
                    "be differentiated."
                )
            value = value.real
        return float(value)

    def _warn_if_upstream_unconverged(self):
        """Warn if any stage of the reference chain stopped short of convergence."""
        for stage in list_method_chain(self.parent_method):
            if getattr(stage, "converged", True) is False:
                logger.log_warning(
                    f"{type(stage).__name__} did not converge at the reference "
                    "geometry; the finite-difference gradient is unreliable."
                )

    def _warn_on_residuals(self):
        """Warn when the measured invariance residuals (see class docstring) are large."""
        for name, residual in (
            ("net force", self.net_force),
            ("net torque", self.net_torque),
        ):
            norm = float(np.linalg.norm(residual))
            if norm > self.residual_tol:
                logger.log_warning(
                    f"Finite-difference residual {name} is {norm:.3e} Eh/Bohr, above "
                    f"residual_tol={self.residual_tol:.3e}. The gradient is likely "
                    "inaccurate at that level: tighten the convergence thresholds of "
                    f"{type(self.parent_method).__name__} or increase step."
                )

    def _warn_on_energy_jumps(self, gradient):
        """
        Warn when a displaced energy is far from the reference energy.

        Over a displacement this small the energy should change by roughly
        ``offset * step * |gradient|``. A much larger change means that
        displacement converged to something else -- a different SCF solution, or
        a different CI root -- in which case the difference quotient straddles a
        discontinuity and the gradient is meaningless rather than merely noisy.
        """
        if not self._displaced_energies:
            return
        offsets, _, _ = central_stencil(self.npoints)
        expected = max(abs(o) for o in offsets) * self.step * np.max(np.abs(gradient))
        observed = float(np.max(np.abs(np.asarray(self._displaced_energies) - self.E)))
        # A generous factor: this should only fire on a qualitative change, not
        # on the second-order curvature the linear estimate ignores.
        threshold = max(_ENERGY_JUMP_FACTOR * expected, _ENERGY_JUMP_FLOOR)
        if observed > threshold:
            logger.log_warning(
                f"A displaced energy differs from the reference by {observed:.3e} Eh, "
                f"far more than the {expected:.3e} Eh expected from the gradient. "
                "Some displacement probably converged to a different SCF solution "
                "or CI root, which invalidates the finite difference."
            )

    def _report_progress(self, done, total):
        self.n_evaluations = done
        logger.log_info2(f"  finite-difference displacement {done}/{total}")

    def _print_start(self, coordinates):
        logger.log_info1("\n==> FINITE-DIFFERENCE NUCLEAR GRADIENT <==")
        logger.log_info1(f"Method: {type(self.parent_method).__name__}")
        logger.log_info1(f"Atoms: {len(coordinates)}")
        logger.log_info1(f"Stencil: {self.npoints}-point central")
        logger.log_info1(f"Step: {self.step:.3e} Bohr")
        logger.log_info1(f"Energy evaluations: {self.npoints * coordinates.size}")
        logger.log_info1(
            f"Reference-geometry orbital projection: "
            f"{'on' if self.project_orbitals else 'off'}"
        )

    def _print_finish(self, gradient):
        logger.log_info1("\nGradient [Eh/Bohr]:")
        logger.log_info1("-" * 52)
        logger.log_info1(f"{'Atom':>5} {'X':>15} {'Y':>15} {'Z':>15}")
        logger.log_info1("-" * 52)
        for atom, row in enumerate(gradient):
            logger.log_info1(f"{atom:>5} {row[0]:15.8f} {row[1]:15.8f} {row[2]:15.8f}")
        logger.log_info1("-" * 52)
        logger.log_info1(f"Norm: {np.linalg.norm(gradient):12.6e} Eh/Bohr")
        # Both vanish for an exact gradient, so they measure the numerical error.
        logger.log_info1(f"Residual net force:  {np.linalg.norm(self.net_force):12.6e}")
        logger.log_info1(
            f"Residual net torque: {np.linalg.norm(self.net_torque):12.6e}"
        )
        logger.log_info1(f"Wall time: {self.wall_time:.2f} s")
