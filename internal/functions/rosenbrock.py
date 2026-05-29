import numpy as np

class RosenbrockFunction:
    """
    Функция Розенброка произвольной размерности (обычно n=2).
    """
    def f(self, x):
        return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2 for i in range(len(x)-1))

    def grad(self, x):
        g = np.zeros_like(x)
        for i in range(len(x)-1):
            g[i] += -400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1.0 - x[i])
            g[i+1] += 200.0 * (x[i+1] - x[i]**2)
        return g

    def hess(self, x):
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n-1):
            H[i, i] += 1200.0 * x[i]**2 - 400.0 * x[i+1] + 2.0
            H[i, i+1] += -400.0 * x[i]
            H[i+1, i] += -400.0 * x[i]
            H[i+1, i+1] += 200.0
        return H