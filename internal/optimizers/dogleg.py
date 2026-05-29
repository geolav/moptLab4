import numpy as np
from internal.utils.result import OptimizationResult


class PowellDogLeg:
    """ Метод Powell's Dog Leg в рамках концепции Trust Region (Доверительной области). """

    def minimize(self, oracle, x0, tol=1e-8, max_iter=1000, delta_max=10.0, delta_init=1.0, eta=0.15):
        x = x0.astype(float)
        path = [x.copy()]
        delta = delta_init

        for k in range(max_iter):
            g = oracle.grad(x)
            if np.linalg.norm(g) <= tol:
                return OptimizationResult(x, oracle.f(x), k, oracle.f_count, oracle.g_count, oracle.h_count,
                                          "Converged", path)

            H = oracle.hess(x)

            # Полный шаг Ньютона
            try:
                pB = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Адаптивный шаг в случае вырожденности
                pB = -g * 0.1

            # Шаг Коши (направление наискорейшего спуска)
            gHg = np.dot(g, H @ g)
            norm_g = np.linalg.norm(g)
            if gHg <= 0:
                pC = -g * (delta / norm_g)
            else:
                pC = - (norm_g ** 2 / gHg) * g

            # Расчет итогового шага вдоль ломаной "Dogleg"
            norm_pB = np.linalg.norm(pB)
            norm_pC = np.linalg.norm(pC)

            if norm_pB <= delta:
                p = pB
            elif norm_pC >= delta:
                p = (delta / norm_pC) * pC
            else:
                # Пересечение с границей доверительной области: ||pC + beta*(pB - pC)|| = delta
                d = pB - pC
                c_dot = np.dot(pC, d)
                norm_d2 = np.dot(d, d)
                discriminant = c_dot ** 2 - norm_d2 * (norm_pC ** 2 - delta ** 2)
                beta = (-c_dot + np.sqrt(discriminant)) / norm_d2
                p = pC + beta * d

            # Проверка качества предсказания модели
            f_current = oracle.f(x)
            actual_reduction = f_current - oracle.f(x + p)
            predicted_reduction = -np.dot(g, p) - 0.5 * np.dot(p, H @ p)

            rho = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0.0

            if rho > eta:
                x = x + p
                path.append(x.copy())

            # Изменение размеров доверительной области
            if rho < 0.25:
                delta *= 0.25
            else:
                if rho > 0.75 and np.linalg.norm(p) == delta:
                    delta = min(2.0 * delta, delta_max)

        return OptimizationResult(x, oracle.f(x), max_iter, oracle.f_count, oracle.g_count, oracle.h_count,
                                  "Max iterations reached", path)