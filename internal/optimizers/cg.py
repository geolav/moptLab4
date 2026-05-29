import numpy as np
from internal.utils.result import OptimizationResult


def backtracking_line_search(oracle, x, p, g, alpha=1.0, rho=0.5, c=1e-4):
    """ Линейный поиск Армихо для нелинейных оптимизаторов """
    f_val = oracle.f(x)
    dot_g_p = np.dot(g, p)
    while oracle.f(x + alpha * p) > f_val + c * alpha * dot_g_p:
        alpha *= rho
        if alpha < 1e-14:
            break
    return alpha


class ConjugateGradientQuad:
    """ Метод сопряженных градиентов для строго квадратичных функций. """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        path = [x.copy()]
        g = oracle.grad(x)
        p = -g

        for k in range(max_iter):
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            A = oracle.hess(x)
            Ap = A @ p
            pAp = np.dot(p, Ap)
            if pAp <= 0:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Break: Non-convex surface direction", path)

            alpha = np.dot(g, g) / pAp
            x_next = x + alpha * p
            path.append(x_next.copy())

            g_next = oracle.grad(x_next)
            beta = np.dot(g_next, g_next) / np.dot(g, g)
            p = -g_next + beta * p

            x, g = x_next, g_next

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)


class NonlinearCGFR:
    """ Метод нелинейных сопряженных градиентов Флетчера–Ривса. """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        path = [x.copy()]
        g = oracle.grad(x)
        p = -g

        for k in range(max_iter):
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            alpha = backtracking_line_search(oracle, x, p, g)
            x = x + alpha * p
            path.append(x.copy())

            g_next = oracle.grad(x)
            beta = np.dot(g_next, g_next) / np.dot(g, g)
            p = -g_next + beta * p
            g = g_next

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)


class NonlinearCGPR:
    """ Метод нелинейных сопряженных градиентов Полака–Рибьера. """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        path = [x.copy()]
        g = oracle.grad(x)
        p = -g

        for k in range(max_iter):
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            alpha = backtracking_line_search(oracle, x, p, g)
            x = x + alpha * p
            path.append(x.copy())

            g_next = oracle.grad(x)
            y = g_next - g
            beta = max(0.0, np.dot(g_next, y) / np.dot(g, g))  # Сброс при потере ортогональности
            p = -g_next + beta * p
            g = g_next

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)