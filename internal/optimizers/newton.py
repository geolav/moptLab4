import numpy as np
from scipy.linalg import cholesky
from internal.utils.result import OptimizationResult
from internal.optimizers.cg import backtracking_line_search


class NewtonCholesky:
    """ Метод Ньютона с использованием разложения Холецкого (без модификаций). """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        path = [x.copy()]

        for k in range(max_iter):
            g = oracle.grad(x)
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            H = oracle.hess(x)
            try:
                # Разложение Холецкого: H = L @ L.T
                L = cholesky(H, lower=True)
                # Решение двух треугольных систем L y = -g и L.T p = y
                y = np.linalg.solve(L, -g)
                p = np.linalg.solve(L.T, y)
            except np.linalg.LinAlgError:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Stopped: Hessian is not Positive Definite", path)

            x = x + p
            path.append(x.copy())

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)


class NewtonDirectionSearch:
    """ Метод Ньютона с выбором направления и оптимизацией шага. """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        path = [x.copy()]

        for k in range(max_iter):
            g = oracle.grad(x)
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            H = oracle.hess(x)

            # Регуляризация матрицы Гессе, если она не является положительно определенной
            w, v = np.linalg.eigh(H)
            if np.min(w) <= 1e-4:
                tau = max(0.0, 1e-3 - np.min(w))
                H_mod = H + tau * np.eye(len(x))
            else:
                H_mod = H

            p = np.linalg.solve(H_mod, -g)

            # Линейный поиск шага для обеспечения глобальной сходимости
            alpha = backtracking_line_search(oracle, x, p, g)
            x = x + alpha * p
            path.append(x.copy())

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)