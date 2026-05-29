import numpy as np
from internal.utils.result import OptimizationResult
from internal.optimizers.cg import backtracking_line_search


class DFPOptimizer:
    """ Квазиньютоновский метод Давидона–Флетчера–Пауэлла. """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        n = len(x)
        path = [x.copy()]
        I = np.eye(n)
        G = I.copy()  # Приближение обратного Гессиана

        g = oracle.grad(x)
        for k in range(max_iter):
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            p = -G @ g
            if np.dot(p, g) >= 0:  # Возвращаемся к градиентному шагу в случае потери положительной определенности
                G = I.copy()
                p = -g

            alpha = backtracking_line_search(oracle, x, p, g)
            x_next = x + alpha * p
            path.append(x_next.copy())

            g_next = oracle.grad(x_next)
            s = x_next - x
            y = g_next - g

            # Формула обновления DFP
            y_G_y = np.dot(y, G @ y)
            s_dot_y = np.dot(s, y)

            if s_dot_y > 1e-10:
                G = G + np.outer(s, s) / s_dot_y - (G @ np.outer(y, y) @ G) / y_G_y

            x, g = x_next, g_next

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)


class BFGSOptimizer:
    """ Квазиньютоновский метод Бройдена–Флетчера–Гольдфарба–Шанно. """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        n = len(x)
        path = [x.copy()]
        I = np.eye(n)
        G = I.copy()

        g = oracle.grad(x)
        for k in range(max_iter):
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            p = -G @ g
            if np.dot(p, g) >= 0:
                G = I.copy()
                p = -g

            alpha = backtracking_line_search(oracle, x, p, g)
            x_next = x + alpha * p
            path.append(x_next.copy())

            g_next = oracle.grad(x_next)
            s = x_next - x
            y = g_next - g

            rho = 1.0 / np.dot(y, s) if np.dot(y, s) != 0 else 0.0
            if rho > 0:
                G = (I - rho * np.outer(s, y)) @ G @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

            x, g = x_next, g_next

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)


class LBFGSOptimizer:
    """ Ограниченный по памяти квазиньютоновский метод L-BFGS. """

    def __init__(self, memory_size=5):
        self.m = memory_size

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000):
        x = x0.astype(float)
        path = [x.copy()]
        history = []  # Хранилище кортежей пар (s_k, y_k)

        g = oracle.grad(x)
        for k in range(max_iter):
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            # Двухэтапная рекурсия (Two-loop recursion)
            q = g.copy()
            alphas = []

            for s_i, y_i, rho_i in reversed(history):
                alpha_i = rho_i * np.dot(s_i, q)
                alphas.append(alpha_i)
                q -= alpha_i * y_i

            if len(history) > 0:
                s_last, y_last, _ = history[-1]
                gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
            else:
                gamma = 1.0

            r = gamma * q
            for (s_i, y_i, rho_i), alpha_i in zip(history, reversed(alphas)):
                beta_i = rho_i * np.dot(y_i, r)
                r += s_i * (alpha_i - beta_i)

            p = -r
            alpha = backtracking_line_search(oracle, x, p, g)
            x_next = x + alpha * p
            path.append(x_next.copy())

            g_next = oracle.grad(x_next)
            s = x_next - x
            y = g_next - g
            ys_dot = np.dot(y, s)

            if ys_dot > 1e-10:
                rho = 1.0 / ys_dot
                history.append((s, y, rho))
                if len(history) > self.m:
                    history.pop(0)

            x, g = x_next, g_next

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)