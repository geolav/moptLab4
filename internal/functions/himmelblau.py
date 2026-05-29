import numpy as np

class HimmelblauFunction:
    """
    Функция Химмельблау (классический двумерный нелинейный тест).
    """
    def f(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def grad(self, x):
        g = np.zeros(2)
        g[0] = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
        g[1] = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
        return g

    def hess(self, x):
        H = np.zeros((2, 2))
        H[0, 0] = 12 * x[0]**2 + 4 * x[1] - 42
        H[0, 1] = 4 * x[0] + 4 * x[1]
        H[1, 0] = H[0, 1]
        H[1, 1] = 4 * x[0] + 12 * x[1]**2 - 26
        return H