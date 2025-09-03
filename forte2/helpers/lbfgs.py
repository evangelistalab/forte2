import numpy as np
from numpy.typing import NDArray
import scipy

from dataclasses import dataclass, field


@dataclass
class LBFGS:
    """
    Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.

    Parameters
    ----------
    print : int, optional, default=1
        Verbosity level for logging. Higher values produce more output.
    m : int, optional, default=6
        The number of vectors to keep in memory for the L-BFGS update.
    epsilon : float, optional, default=1.0e-5
        Convergence threshold to terminate the minimization: |g| < ε * max(1, |x|).
    maxiter : int, optional, default=20
        Maximum number of iterations for the optimization.
    h0_freq : int, optional, default=0
        Frequency of updating the diagonal Hessian. If 0, it is computed only for the
        initial iteration. If < 0, uses the adaptive inverse Hessian.
    maxiter_linesearch : int, optional, default=5
        Maximum number of trials for line search to find the optimal step length.
    max_dir : float, optional, default=1.0e15
        Maximum absolute value allowed in the direction vector.
    line_search_condition : str, optional, default='strong_wolfe'
        Condition to terminate line search backtracking. Options are 'armijo', 'wolfe',
        or 'strong_wolfe'.
    step_length_method : str, optional, default='line_bracketing_zoom'
        Method to determine step lengths. Options are 'max_correction',
        'line_backtracking', or 'line_bracketing_zoom'.
    min_step : float, optional, default=1.0e-15
        Minimum step length allowed during line search.
    max_step : float, optional, default=1.0e15
        Maximum step length allowed during line search.
    c1 : float, optional, default=1.0e-4
        Parameter for the Armijo condition in line search.
    c2 : float, optional, default=0.9
        Parameter for the Wolfe curvature condition in line search.

    Notes
    -----
    Translated into Python from Forte v1: https://github.com/evangelistalab/forte/tree/main/forte/helpers/lbfgs
    For implementation details, see Wikipedia https://en.wikipedia.org/wiki/Limited-memory_BFGS
    and Numerical Optimization 2nd Ed. by Jorge Nocedal and Stephen J. Wright
    """

    ### L-BFGS parameters
    print: int = 1
    m: int = 6
    epsilon: float = 1.0e-5
    maxiter: int = 20
    h0_freq: int = 0
    maxiter_linesearch: int = 5
    max_dir: float = 1.0e15
    line_search_condition: str = "strong_wolfe"
    step_length_method: str = "line_bracketing_zoom"
    min_step: float = 1.0e-15
    max_step: float = 1.0e15
    c1: float = 1.0e-4
    c2: float = 0.9

    ### Non-init params
    converged: bool = field(default=False, init=False)
    iter: int = field(default=0, init=False)
    # The correction (moving direction) vector
    p: NDArray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        self._check_param()

    def _check_param(self):
        assert self.line_search_condition.lower() in [
            "armijo",
            "wolfe",
            "strong_wolfe",
        ], "line_search_condition must be one of 'armijo', 'wolfe', or 'strong_wolfe'"
        assert self.step_length_method.lower() in [
            "max_correction",
            "line_backtracking",
            "line_bracketing_zoom",
        ], "step_length_method must be one of 'max_correction', 'line_backtracking', or 'line_bracketing_zoom'"
        self.line_search_condition = self.line_search_condition.lower()
        self.step_length_method = self.step_length_method.lower()

        if self.m <= 0:
            raise ValueError("Size of L-BFGS history (m) must > 0")
        if self.epsilon <= 0:
            raise ValueError("Convergence threshold (epsilon) must > 0")
        if self.maxiter <= 0:
            raise ValueError("Max number of iterations (maxiter) must > 0")
        if self.maxiter_linesearch <= 0:
            raise ValueError(
                "Max iterations for line search (maxiter_linesearch) must > 0"
            )
        if self.max_dir < 0:
            raise ValueError(
                "Max absolute value in direction vector (max_dir) must > 0"
            )
        if self.min_step < 0:
            raise ValueError("Minimum step length (min_step) must > 0")
        if self.max_step < self.min_step:
            raise ValueError("Maximum step length (max_step) must > min_step")
        if not (0 < self.c1 < 0.5):
            raise ValueError("Parameter c1 must lie in (0, 0.5)")
        if not (self.c1 < self.c2 < 1):
            raise ValueError("Parameter c2 must lie in (c1, 1.0)")

    def minimize(self, obj, x):
        """
        The minimization for the objective function

        Parameters
        ----------
            obj : object
                Target function to minimize, it should should be encapsulated in a class that has the following methods:
                ``fx, g = obj.evaluate(x, g, do_g=True)`` where gradient ``g`` is modified by the function,
                ``fx`` is the function return value, and ``g`` is computed when ``do_g==True``.
                If diagonal Hessian is specified, ``h0 = obj.hess_diag(x)`` should be available.
            x : NDArray
                The initial value of ``x`` as input, the final value of ``x`` as output.

        Returns
        -------
            fx : float
                The function value at optimized ``x``.
        """
        self.g = np.zeros_like(x)
        fx, self.g = obj.evaluate(x, self.g)
        x_norm = np.linalg.norm(x)
        g_norm = np.linalg.norm(self.g)

        if g_norm <= self.epsilon * max(1.0, x_norm):
            self.converged = True
            self.iter = 0
            return fx

        def compute_h0(x_vec):
            self.h0 = obj.hess_diag(x_vec)
            if self.print > 3:
                print(f"Diagonal Hessian at Iter. {self.iter}")
                print(self.h0)

        self._reset()
        self.p = np.zeros_like(x)
        self.x_last = x.copy()
        self.g_last = self.g.copy()
        if self.h0_freq >= 0:
            self.h0 = np.zeros_like(x)
            compute_h0(x)

        self.converged = False
        while self.iter < self.maxiter:
            if self.iter != 0 and self.h0_freq > 0:
                if self.iter % self.h0_freq == 0:
                    compute_h0(x)

            self._update()

            step = 1.0
            fx, step = self._next_step(obj, x, fx, step)

            g_norm = np.linalg.norm(self.g)
            if self.print > 2:
                print(
                    f"    L-BFGS Iter:{self.iter + 1:3d}; fx = {fx:20.15f}; g_norm = {g_norm:12.6e}; step = {step:9.3e}"
                )

            self.iter += 1
            if self.iter == self.maxiter:
                break

            x_norm = np.linalg.norm(x)
            if g_norm <= self.epsilon * max(1.0, x_norm):
                self.converged = True
                break

            s = x - self.x_last
            y = self.g - self.g_last
            rho = np.dot(y, s)

            if rho > 0:
                iter_idx = self.iter - self.iter_shift_
                index = (iter_idx - 1) % self.m
                if iter_idx <= self.m:
                    self.s[index] = np.zeros_like(x)
                    self.y[index] = np.zeros_like(x)
                self.s[index][:] = s
                self.y[index][:] = y
                self.rho[index] = 1.0 / rho
                self.x_last[:] = x
                self.g_last[:] = self.g
            else:
                self.iter_shift_ += 1
                if self.print > 1:
                    print("  L-BFGS Warning: Skip this vector due to negative rho")

        if not self.converged and self.print > 2:
            print(f"  L-BFGS Warning: No convergence in {self.iter} iterations")

        return fx

    def _update(self):
        self.p[:] = self.g
        m = min(self.iter - self.iter_shift_, self.m)
        end = (self.iter - self.iter_shift_ - 1) % m if m else 0

        for k in range(m):
            i = (end - k + m) % m
            self.alpha[i] = self.rho[i] * np.dot(self.s[i], self.p)
            self.p -= self.alpha[i] * self.y[i]

        self._apply_h0(self.p)

        for k in range(m):
            i = (end + k + 1) % m
            beta = self.rho[i] * np.dot(self.y[i], self.p)
            self.p += (self.alpha[i] - beta) * self.s[i]

        self.p *= -1.0
        if self.print > 3:
            print(self.p)

    def _next_step(self, foo, x, fx, step):
        match self.step_length_method:
            case "max_correction":
                return self._scale_direction_vector(foo, x, fx, step)
            case "line_backtracking":
                return self._line_search_backtracking(foo, x, fx, step)
            case "line_bracketing_zoom":
                return self._line_search_bracketing_zoom(foo, x, fx, step)

    def _scale_direction_vector(self, func, x, fx, step):
        p_max = np.max(np.abs(self.p))
        step = min(1.0, self.max_dir / p_max) if p_max > self.max_dir else 1.0
        x += step * self.p
        do_grad = self.iter - self.iter_shift_ + 1 < self.maxiter
        fx, self.g = func.evaluate(x, self.g, do_grad)
        return fx, step

    def _line_search_backtracking(self, func, x, fx, step):
        dg0 = np.dot(self.g, self.p)
        fx0 = fx
        x0 = x.copy()

        if dg0 >= 0:
            print("  Warning: Direction increases the energy. Reset L-BFGS.")
            self._reset()

        for i in range(self.maxiter_linesearch + 1):
            x[:] = x0 + step * self.p
            fx, self.g = func.evaluate(x, self.g)

            if i == self.maxiter_linesearch:
                break

            if fx - fx0 > self.c1 * dg0 * step:
                step *= 0.5
            else:
                if self.line_search_condition == "armijo":
                    break
                dg = np.dot(self.g, self.p)
                if dg < self.c2 * dg0:
                    step *= 2.05
                else:
                    if self.line_search_condition == "wolfe":
                        break
                    if abs(dg) > -self.c2 * dg0:
                        step *= 0.5
                    else:
                        break
            if step > self.max_step:
                if self.print > 1:
                    print("    Step length > max allowed value. Stopped line search.")
                step = self.max_step
                break
            if step < self.min_step:
                if self.print > 1:
                    print("    Step length < min allowed value. Stopped line search.")
                step = self.min_step
                break
        return fx, step

    def _line_search_bracketing_zoom(self, func, x, fx, step):
        dg0 = np.dot(self.g, self.p)
        fx0 = fx
        x0 = x.copy()

        if dg0 >= 0:
            print("  Warning: Direction increases the energy. Reset L-BFGS.")
            self._reset()

        w1 = self.c1 * dg0
        w2 = -self.c2 * dg0
        fx_low = fx0
        step_low, step_high = 0.0, 0.0

        for i in range(self.maxiter_linesearch):
            x[:] = x0 + step * self.p
            fx, self.g = func.evaluate(x, self.g)

            if fx - fx0 > w1 * step or (fx >= fx_low and i > 0):
                step_high = step
                break

            dg = np.dot(self.g, self.p)
            if abs(dg) <= w2:
                if self.print > 3:
                    print(f"    Optimal step length from bracketing stage: {step:.15f}")
                return fx, step

            step_high = step_low
            fx_low = fx
            step_low = step

            if dg >= 0:
                break
            step *= 2.0

        if self.print > 3:
            print(
                f"    Step lengths after bracketing stage: low = {step_low:.10f}, high = {step_high:.10f}"
            )

        for i in range(self.maxiter_linesearch + 1):
            step = 0.5 * (step_low + step_high)
            x[:] = x0 + step * self.p
            fx, self.g = func.evaluate(x, self.g)

            if i == self.maxiter_linesearch:
                break

            if fx - fx0 > w1 * step or fx >= fx_low:
                step_high = step
            else:
                dg = np.dot(self.g, self.p)
                if abs(dg) <= w2:
                    if self.print > 3:
                        print(
                            f"    Optimal step length from zooming stage: {step:.15f}"
                        )
                    break
                if dg * (step_high - step_low) >= 0:
                    step_high = step_low
                step_low = step
                fx_low = fx

        if self.print > 3:
            print(
                f"    Step lengths after zooming stage: low = {step_low:.10f}, high = {step_high:.10f}"
            )

        if step > self.max_step:
            if self.print > 1:
                print("    Step length > max allowed value. Use max allowed value.")
            step = self.max_step
        if step < self.min_step:
            if self.print > 1:
                print("    Step length < min allowed value. Use min allowed value.")
            step = self.min_step
        return fx, step

    def _apply_h0(self, q):
        if self.h0_freq < 0:
            gamma = self._compute_gamma()
            if self.print > 3:
                print(f"    gamma for H0: {gamma:.15f}")
            q *= gamma
        else:
            vh = self.h0
            mask = np.abs(vh) > 1.0e-12
            q[mask] /= vh[mask]
            if self.print > 1 and not np.all(mask):
                print("    Zero diagonal Hessian element")

    def _compute_gamma(self):
        if self.iter - self.iter_shift_ == 0:
            return 1.0
        end = (self.iter - self.iter_shift_ - 1) % min(
            self.iter - self.iter_shift_, self.m
        )
        return np.dot(self.s[end], self.y[end]) / np.dot(self.y[end], self.y[end])

    def _resize(self, m):
        self.y = [np.array([]) for _ in range(m)]
        self.s = [np.array([]) for _ in range(m)]
        self.alpha = np.zeros(m)
        self.rho = np.zeros(m)

    def _reset(self):
        self._resize(self.m)
        self.iter = 0
        self.iter_shift_ = 0


@dataclass
class LBFGS_scipy:
    """
    A wrapper for the SciPy L-BFGS optimization. For debug use only.
    """

    epsilon: float = 1.0e-5
    maxiter: int = 20
    c1: float = 1.0e-4
    c2: float = 0.9

    ### Unused parameters to match custom implementation
    print: int = 1
    m: int = 6
    h0_freq: int = 0
    maxiter_linesearch: int = 5
    max_dir: float = 1.0e15
    line_search_condition: str = "strong_wolfe"
    step_length_method: str = "line_bracketing_zoom"
    min_step: float = 1.0e-15
    max_step: float = 1.0e15

    def minimize(self, obj, x):
        fun = lambda x: obj.evaluate(x, None, do_g=False)[0]
        jac = lambda x: obj.evaluate(x, np.zeros_like(x), do_g=True)[1]
        _ = fun(x)
        _ = jac(x)
        hess_diag = obj.hess_diag(x)
        hess_inv0 = np.diag(
            np.divide(
                1.0,
                hess_diag,
                out=np.zeros_like(hess_diag),
                where=np.abs(hess_diag) >= 1e-12,
            )
        )
        res = scipy.optimize.minimize(
            fun,
            x,
            jac=jac,
            method="BFGS",
            options={
                "maxiter": self.maxiter,
                "c1": self.c1,
                "c2": self.c2,
                "hess_inv0": hess_inv0,
            },
        )
        self.g = res.x.copy()
        self.converged = res.success
        self.iter = res.nit
        return res.fun
