import numpy as np
from scipy.optimize import minimize
from internal.utils.result import OptimizationResult


class ScipyNewtonCG:
    """ Обертка над библиотечным методом Newton-CG из scipy. """

    def minimize(self, oracle, x0, tol=1e-8):
        # Логирование пути через callback
        path = [x0.copy()]

        def callback(xk):
            path.append(xk.copy())

        res = minimize(
            fun=oracle.f,
            x0=x0,
            method='Newton-CG',
            jac=oracle.grad,
            hess=oracle.hess,
            tol=tol,
            callback=callback
        )
        return OptimizationResult(res.x, res.fun, res.nit, oracle.f_count, oracle.g_count, oracle.h_count, res.message,
                                  path)