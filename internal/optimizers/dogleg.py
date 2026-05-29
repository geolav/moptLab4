import numpy as np
from internal.utils.result import OptimizationResult


class PowellDogLeg:
    def __init__(self, max_iter=1000, max_tr=10.0, initial_tr=1.0):
        self.max_iter = max_iter
        self.max_tr = max_tr
        self.initial_tr = initial_tr

    def minimize(self, oracle, x0, tol=1e-5):
        x = np.array(x0, dtype=float)
        tr_radius = self.initial_tr
        path = [x.copy()]

        status = "Max iterations reached"
        iters = self.max_iter  # Дефолтное значение

        for i in range(self.max_iter):
            g = oracle.grad(x)
            norm_g = np.linalg.norm(g)

            if norm_g < tol:
                status = "Converged"
                iters = i  # Записываем точное число пройденных итераций
                break

            B = oracle.hess(x)

            # Точка Коши (Cauchy Point)
            gBg = np.dot(g, np.dot(B, g))
            if gBg <= 0:
                tau = 1.0
            else:
                tau = min(1.0, norm_g ** 3 / (tr_radius * gBg))

            pC = -tau * (tr_radius / norm_g) * g
            norm_pC = np.linalg.norm(pC)

            # Ньютоновский шаг
            try:
                pN = np.linalg.solve(B, -g)
            except np.linalg.LinAlgError:
                pN = pC  # Фолбэк, если матрица вырождена

            norm_pN = np.linalg.norm(pN)

            # Выбор шага DogLeg
            if norm_pN <= tr_radius:
                p = pN
            elif norm_pC >= tr_radius:
                if norm_pC < 1e-12:
                    p = np.zeros_like(pC)
                else:
                    p = (tr_radius / norm_pC) * pC
            else:
                pB_pC = pN - pC
                a = np.dot(pB_pC, pB_pC)
                b = 2 * np.dot(pC, pB_pC)
                c = np.dot(pC, pC) - tr_radius ** 2

                discriminant = b ** 2 - 4 * a * c
                if discriminant < 0 or a == 0:
                    p = pC
                else:
                    tau_intersect = (-b + np.sqrt(discriminant)) / (2 * a)
                    p = pC + tau_intersect * pB_pC

            # Оценка шага (Trust Region Update)
            m_p = np.dot(g, p) + 0.5 * np.dot(p, np.dot(B, p))

            # ИСПРАВЛЕНИЕ: Используем .f(...) вместо .func(...) для совместимости с оракулом
            f_x = oracle.f(x)
            f_new = oracle.f(x + p)

            actual_reduction = f_x - f_new
            predicted_reduction = -m_p

            if predicted_reduction < 1e-12:
                rho = 0
            else:
                rho = actual_reduction / predicted_reduction

            if rho < 0.25:
                tr_radius *= 0.25
            else:
                if rho > 0.75 and np.abs(np.linalg.norm(p) - tr_radius) < 1e-8:
                    tr_radius = min(2.0 * tr_radius, self.max_tr)

            if rho > 0:
                x = x + p
                path.append(x.copy())

        # ИСПРАВЛЕНИЕ: Передаем строго все поля, требуемые датаклассом OptimizationResult,
        # включая счетчики вызовов функций (f_evals, g_evals, h_evals) для корректного вывода таблицы.
        return OptimizationResult(
            x_opt=x,
            f_opt=oracle.f(x),
            iters=iters,
            f_evals=oracle.f_count,
            g_evals=oracle.g_count,
            h_evals=oracle.h_count,
            status=status,
            path=path
        )