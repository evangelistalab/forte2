import numpy as np

from scipy.optimize import minimize
from forte2.helpers import LBFGS
from forte2.helpers.comparisons import approx


class Rosenbrock:
    def evaluate(x):
        r"""
        Rosenbrock function

        See https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#unconstrained-minimization-of-multivariate-scalar-functions-minimize,
        Here we have the dummy index i shift up by 1.

        f(\bvec{x}) = \sum_{i=1}^{N-1}[100(x_{i+1}-x_i^2)^2+(1-x_i)^2]
        """
        t1 = x[1:] - x[:-1] ** 2
        t2 = 1.0 - x[:-1]
        return np.sum(100.0 * np.dot(t1, t1) + np.dot(t2, t2))

    def gradient(x):
        grad = np.zeros_like(x)
        grad[1:] = -(1.0 - x[1:]) + 200 * (x[1:] - x[:-1] ** 2)
        grad[:-1] -= 400 * x[:-1] * (x[1:] - x[:-1] ** 2)
        return grad

    def diagonal_hessian(x):
        h0 = 202 + 400 * x**2
        h0[:-1] -= 400 * x[:-1]
        return h0


class RosenbrockComplex:
    def evaluate(x):
        l = len(x) // 2
        z = x[:l] + 1.0j * x[l:]
        t1 = z[1:] - z[:-1] ** 2
        t2 = 1.0 - z[:-1]
        return np.sum(100.0 * np.dot(t1.conj(), t1).real + np.dot(t2.conj(), t2).real)

    def gradient(x):
        """
        One of the Wirtinger gradients, df/dz*, is used for descent.
        See Ch. 4 of https://mediatum.ub.tum.de/doc/631019/631019.pdf
        df/dz*_i = 1/2 (df/dx_i + i df/dy_i), so 2 * [Re(df/dz*_i), Im(df/dz*_i)] = [df/dx_i, df/dy_i]
        """
        l = len(x) // 2
        z = x[:l] + 1.0j * x[l:]
        grad = np.zeros(l, dtype=np.complex128)
        grad[1:] = -(1.0 - z[1:]) + 100 * (z[1:] - z[:-1] ** 2)
        grad[:-1] -= 200 * z[:-1].conj() * (z[1:] - z[:-1] ** 2)
        g = 2 * np.concatenate([grad.real, grad.imag])
        return g

    def diagonal_hessian(x):
        """
        Same logic, if we want [d^2f/dx_i^2, d^2f/dy_i^2],
        we need to compute 2 * [d^2f/(dz_i dz^*_i) + Re(d^2f/dz^*_i^2), d^2f/(dz_i dz^*_i) - Re(d^2f/dz^*_i^2)]
        """
        l = len(x) // 2
        z = x[:l] + 1.0j * x[l:]
        hess = 400 * np.dot(z.conj(), z).real + 101
        squared_term = 200 * np.real(z**2)
        squared_term[:-1] -= 200 * np.real(z[:-1])
        return 2 * np.concatenate([hess - squared_term, hess + squared_term])


class RosenbrockWrapper:
    def evaluate(self, x, g, do_g=True):
        fx = Rosenbrock.evaluate(x)
        if do_g:
            g = Rosenbrock.gradient(x)
        return fx, g

    def hess_diag(self, x):
        return Rosenbrock.diagonal_hessian(x)


class RosenbrockComplexWrapper:
    def evaluate(self, x, g, do_g=True):
        fx = RosenbrockComplex.evaluate(x)
        if do_g:
            g = RosenbrockComplex.gradient(x)
        return fx, g

    def hess_diag(self, x):
        return RosenbrockComplex.diagonal_hessian(x)


def test_lbfgs_rosenbrock():
    n = 10
    h0_freq = 0
    lbfgs_solver = LBFGS()
    lbfgs_solver.epsilon = 1.0e-6
    lbfgs_solver.maxiter = 200
    lbfgs_solver.h0_freq = h0_freq
    lbfgs_solver.print = 2

    func = RosenbrockWrapper()
    x = np.ones(n) * 0.1
    fx = lbfgs_solver.minimize(func, x)

    x0 = np.ones(n) * 0.1
    hess = Rosenbrock.diagonal_hessian(x)
    hess_inv0 = np.diag(
        np.divide(1.0, hess, out=np.zeros_like(hess), where=np.abs(hess) >= 1e-12)
    )
    res = minimize(
        Rosenbrock.evaluate,
        x0,
        jac=Rosenbrock.gradient,
        method="BFGS",
        options={"gtol": 1e-6, "disp": True, "maxiter": 500, "hess_inv0": hess_inv0},
    )

    assert np.linalg.norm(x - res.x) < 1e-6
    assert fx == approx(res.fun)


def test_lbfgs_rosenbrock_complex():
    n = 20
    h0_freq = 0
    lbfgs_solver = LBFGS()
    lbfgs_solver.epsilon = 1.0e-6
    lbfgs_solver.maxiter = 200
    lbfgs_solver.h0_freq = h0_freq
    lbfgs_solver.print = 3
    func = RosenbrockComplexWrapper()
    x = np.ones(n) * 0.1
    fx = lbfgs_solver.minimize(func, x)

    hess = RosenbrockComplex.diagonal_hessian(x)
    hess_inv0 = np.diag(
        np.divide(1.0, hess, out=np.zeros_like(hess), where=np.abs(hess) >= 1e-12)
    )
    x0 = np.ones(20) * 0.1
    res = minimize(
        RosenbrockComplex.evaluate,
        x0,
        jac=RosenbrockComplex.gradient,
        method="BFGS",
        options={"gtol": 1e-6, "disp": True, "maxiter": 500, "hess_inv0": hess_inv0},
    )

    assert np.linalg.norm(x - res.x) < 1e-6
    assert fx == approx(res.fun)
