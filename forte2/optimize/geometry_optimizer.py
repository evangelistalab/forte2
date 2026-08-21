from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from forte2.base_classes import Method
from forte2.base_classes.rebuild import (
    rebind_method_chain,
    rebuild_method_chain,
    project_scf_guess,
    snapshot_orbitals,
)
from forte2.data import Z_TO_ATOM_SYMBOL
from forte2.helpers import LBFGS, logger
from forte2.helpers.table import AsciiTable
from forte2.system import System


@dataclass
class GeometryOptimizer(Method):
    """
    Cartesian geometry optimizer using L-BFGS and method analytic gradients.

    Parameters
    ----------
    method_factory : Callable, optional
        Callable taking a ``System`` and returning a method object that provides
        ``run()``, ``E``, and ``gradient()``. This is retained as a valid alternative.
        The preferred API call is ``GeometryOptimizer(...)(method).run()``.
    maxiter : int, optional, default=50
        Maximum L-BFGS iterations.
    g_tol : float, optional, default=1.0e-4
        L-BFGS gradient convergence threshold.
    max_step : float, optional, default=1.0
        Maximum line-search step length.
    project_orbitals : bool, optional, default=True
        If True, project the occupied orbitals from the previous evaluated
        geometry into the AO basis of the next geometry and use them as the SCF
        initial guess.
    lbfgs_kwargs : dict, optional
        Additional keyword arguments forwarded to ``LBFGS``.
    """

    method_factory: Callable | None = None
    maxiter: int = 50
    g_tol: float = 1.0e-4
    max_step: float = 1.0
    project_orbitals: bool = True
    lbfgs_kwargs: dict = field(default_factory=dict)

    converged: bool = field(default=False, init=False)
    iter: int = field(default=0, init=False)
    E: float | None = field(default=None, init=False)
    gradient: np.ndarray | None = field(default=None, init=False)
    history: list[tuple] = field(default_factory=list, init=False)
    coordinates: np.ndarray | None = field(default=None, init=False)
    system: System | None = field(default=None, init=False)
    method: object | None = field(default=None, init=False)

    def __post_init__(self):
        self.requires = {"system", "mos"}
        self.requires_attrs.update({"gradient": None})

    def __call__(self, method):
        """
        Attach the optimizer to an upstream method.

        The upstream method supplies both the initial ``System`` and the method
        configuration used to rebuild a fresh method chain at each geometry.
        """
        self._register_parent_method(method)
        return self

    def run(self, system=None):
        """
        Optimize a molecular geometry.

        Parameters
        ----------
        system : System, optional
            Initial molecular system. Required only when ``method_factory`` is
            used directly instead of the upstream-method API.

        Returns
        -------
        GeometryOptimizer
            The executed optimizer object.
        """
        objective, x = self._build_objective(system)
        self._print_start(objective)

        # Suppress the printing from the objective function
        current_verbosity = logger.get_verbosity_level()
        logger.set_verbosity_level(max(current_verbosity - 1, 0))

        optimizer = LBFGS(
            epsilon=self.g_tol,
            maxiter=self.maxiter,
            max_step=self.max_step,
            warn_if_not_converged=True,
            **self.lbfgs_kwargs,
        )
        self.E = optimizer.minimize(objective, x)
        objective.ensure(x, need_gradient=True)

        self.executed = True
        self.converged = optimizer.converged
        self.iter = optimizer.iter
        self.system = objective.system
        self.method = objective.method
        self.E = objective.E
        self.gradient = objective.g.reshape(-1, 3).copy()
        self.history = objective.history.copy()
        self.coordinates = x.reshape(-1, 3).copy()

        logger.set_verbosity_level(current_verbosity)

        self._print_finish()

        return self

    def _build_objective(self, system):
        if self.parent_method is not None:
            if system is not None and system is not self.parent_method.system:
                raise ValueError(
                    "Do not pass a separate system when GeometryOptimizer is already "
                    "attached to an upstream method."
                )
            if not self.parent_method.executed:
                self.parent_method.run()

            system = self.parent_method.system
            template_method = self.parent_method
            # scratch_chain is only built once and then reused for every iteration
            scratch_chain = rebuild_method_chain(template_method, system)

            def method_builder(new_system):
                return rebind_method_chain(scratch_chain, new_system)

            objective = _GeometryObjective(
                system,
                method_builder,
                project_orbitals=self.project_orbitals,
                seed_method=self.parent_method,
            )
        else:
            if self.method_factory is None:
                raise ValueError(
                    "Attach the optimizer to a method with "
                    "GeometryOptimizer(...)(method).run(), or provide "
                    "method_factory and pass a system to run(system)."
                )
            if system is None:
                raise ValueError(
                    "system is required when GeometryOptimizer is used with "
                    "method_factory."
                )
            objective = _GeometryObjective(
                system,
                self.method_factory,
                project_orbitals=self.project_orbitals,
            )

        x = np.asarray(system.atomic_positions, dtype=float).reshape(-1).copy()
        return objective, x

    def _print_start(self, objective):
        method_name = _method_name(self.parent_method)
        if method_name is None and objective.method is not None:
            method_name = _method_name(objective.method)
        if method_name is None:
            method_name = "method_factory"

        logger.log_info1("\n==> CARTESIAN GEOMETRY OPTIMIZATION <==")
        logger.log_info1(f"Method: {method_name}")
        logger.log_info1(f"Atoms: {len(objective.atomic_numbers)}")
        logger.log_info1("Optimizer: L-BFGS")
        logger.log_info1(f"Max iterations: {self.maxiter}")
        logger.log_info1(f"Gradient threshold: {self.g_tol:.3e}")
        logger.log_info1(
            f"Previous-geometry orbital projection: {'on' if self.project_orbitals else 'off'}"
        )

    def _print_finish(self):
        gradient_norm = np.linalg.norm(self.gradient)
        gradient_max = np.max(np.abs(self.gradient)) if self.gradient.size else 0.0
        status = "converged" if self.converged else "not converged"

        self._print_history_table()
        logger.log_info1(
            f"\nGeometry optimization {status} in {self.iter} L-BFGS iterations."
        )
        logger.log_info1(f"Final energy: {self.E:20.12f} Eh")
        logger.log_info1(f"Final gradient norm: {gradient_norm:12.6e} Eh/Bohr")
        logger.log_info1(f"Final max gradient:  {gradient_max:12.6e} Eh/Bohr")
        logger.log_info1("\nFinal Cartesian coordinates [Bohr]:")
        logger.log_info1("-" * 55)
        logger.log_info1(f"{'Atom':<6} {'X':>15} {'Y':>15} {'Z':>15}")
        logger.log_info1("-" * 55)
        for atomic_number, xyz in zip(self.system.atomic_charges, self.coordinates):
            symbol = Z_TO_ATOM_SYMBOL[int(atomic_number)]
            logger.log_info1(
                f"{symbol:<6} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}"
            )
        logger.log_info1("-" * 55)

    def _print_history_table(self):
        if not self.history:
            return

        table = AsciiTable(
            columns=["Eval", "Energy [Eh]", "dE", "|grad|", "max|grad|", "step"],
            formats=[
                "{:>5d}",
                "{:>20.12f}",
                "{:>12}",
                "{:>12.4e}",
                "{:>12.4e}",
                "{:>10}",
            ],
        )

        logger.log_info1("\nGeometry optimization trajectory:")
        logger.log_info1(table.header())
        for row in self.history:
            logger.log_info1(table.row(*row))
        logger.log_info1(table.footer())


class _GeometryObjective:
    """LBFGS objective that rebuilds a system at every Cartesian point."""

    def __init__(
        self,
        template_system,
        method_builder,
        project_orbitals=True,
        seed_method=None,
    ):
        self.method_builder = method_builder
        self.project_orbitals = project_orbitals
        self.previous_method = seed_method
        self.template_system = template_system
        self.atomic_numbers = np.asarray(template_system.atomic_charges, dtype=int)

        self.x = None
        self.system = None
        self.method = None
        self.E = None
        self.g = None
        self.eval_count = 0
        self.logged_x = None
        self.logged_E = None
        self.history = []

        if seed_method is not None:
            self.x = np.asarray(
                seed_method.system.atomic_positions, dtype=float
            ).reshape(-1)
            self.system = seed_method.system
            self.method = seed_method
            self.E = float(seed_method.E)

    def evaluate(self, x):
        self.ensure(x, need_gradient=False)
        return self.E

    def gradient(self, x):
        self.ensure(x, need_gradient=True)
        return self.g.copy()

    def hess_diag(self, x):
        return np.ones_like(x)

    def ensure(self, x, need_gradient=False):
        if self._cache_matches(x):
            if need_gradient and self.g is None:
                self.g = np.asarray(self.method.gradient(), dtype=float).reshape(-1)
                self._record_progress()
            return

        # Snapshot the projection source before method_builder runs: once the
        # chain is reused in place (rebind_method_chain), self.previous_method
        # and the about-to-be-rebuilt self.method are the same live object, so
        # its orbitals must be captured now, not read back off it afterwards.
        projection_source = None
        if self.project_orbitals and self.previous_method is not None:
            projection_source = snapshot_orbitals(self.previous_method)

        self.x = np.asarray(x, dtype=float).copy()
        self.system = self.template_system.with_geometry(self.x.reshape(-1, 3))
        self.method = self.method_builder(self.system)
        if projection_source is not None:
            project_scf_guess(projection_source, self.method)
        if not self.method.executed:
            self.method.run()
        self.E = float(self.method.E)
        self.g = None
        self.previous_method = self.method

        if need_gradient:
            self.g = np.asarray(self.method.gradient(), dtype=float).reshape(-1)
            self._record_progress()

    def _cache_matches(self, x):
        return self.x is not None and np.array_equal(self.x, np.asarray(x, dtype=float))

    def _record_progress(self):
        if self.g is None or self.E is None or self.x is None:
            return
        if self.logged_x is not None and np.array_equal(self.logged_x, self.x):
            return

        dE = None if self.logged_E is None else self.E - self.logged_E
        step = None
        if self.logged_x is not None:
            step = np.linalg.norm(self.x - self.logged_x)

        self.eval_count += 1
        self.history.append(
            (
                self.eval_count,
                self.E,
                _format_float(dE),
                np.linalg.norm(self.g),
                np.max(np.abs(self.g)),
                _format_float(step),
            )
        )
        self.logged_x = self.x.copy()
        self.logged_E = self.E


def _method_name(method):
    if method is None:
        return None
    return method._scf_type() if hasattr(method, "_scf_type") else type(method).__name__


def _format_float(value):
    if value is None:
        return "-"
    return f"{value:.4e}"
