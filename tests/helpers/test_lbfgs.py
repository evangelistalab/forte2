import numpy as np
import scipy
from scipy.optimize import minimize
from forte2.helpers import LBFGS
from forte2.helpers.comparisons import approx


class ROSENBROCK:
    def __init__(self, n):
        if n % 2 != 0:
            raise ValueError("Invalid size for Rosenbrock. Please use even number.")
        self.n = n

    def evaluate(self, x, g, do_g=True):
        fx = 0.0
        for i in range(0, self.n, 2):
            xi = x[i]
            t1 = xi - 1.0
            t2 = 10 * (xi * xi - x[i + 1])
            if do_g:
                g[i + 1] = -20 * t2
                g[i] = 2.0 * (t1 - g[i + 1] * xi)
            fx += t1 * t1 + t2 * t2
        return fx, g

    def hess_diag(self, x):
        h0 = np.zeros_like(x)

        for i in range(0, self.n, 2):
            xi = x[i]
            t2 = 10 * (xi * xi - x[i + 1])
            h0[i + 1] = 200
            h0[i] = 2.0 * (1 + 400 * xi * xi + 20 * t2)
        return h0


def scipy_minimize():
    def rosenbrock(x):
        # f(x) = sum_{i=0}^{n/2 - 1} [ 100 * (x_{2i}^2 - x_{2i + 1})^2 + (x_{2i} - 1)^2 ]
        x_even = x[::2]
        x_odd = x[1::2]
        return sum(100.0 * (x_even**2 - x_odd) ** 2 + (x_even - 1) ** 2)

    def rosenbrock_grad(x):
        x_even = x[::2]
        x_odd = x[1::2]
        t1 = x_even - 1
        t2 = 10 * (x_even**2 - x_odd)

        g = np.zeros(x.shape)
        g[1::2] = -20 * t2
        g[::2] = 2.0 * (t1 - g[1::2] * x_even)
        return g

    x0 = np.zeros(10)
    res = minimize(
        rosenbrock,
        x0,
        method="BFGS",
        jac=rosenbrock_grad,
        options={"gtol": 1e-6, "disp": True},
    )

    return res.x, res.fun


def test_lbfgs_rosenbrock():
    n = 10
    h0_freq = 0
    lbfgs_solver = LBFGS()
    lbfgs_solver.epsilon = 1.0e-6
    lbfgs_solver.maxiter = 100
    lbfgs_solver.h0_freq = h0_freq
    lbfgs_solver.print = 2

    rosenbrock = ROSENBROCK(n)
    x = np.zeros(n)

    fx = lbfgs_solver.minimize(rosenbrock, x)
    x_scipy, fx_scipy = scipy_minimize()

    assert np.linalg.norm(x - x_scipy) < 1e-6
    assert fx == approx(fx_scipy)


if __name__ == "__main__":
    test_lbfgs_rosenbrock()
