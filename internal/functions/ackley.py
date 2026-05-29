import numpy as np

class AckleyFunction:
    """
    Функция Экли (n-мерная функция с множеством локальных экстремумов).
    """
    def f(self, x):
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2.0 * np.pi * x))
        return -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

    def grad(self, x, eps=1e-7):
        # Численный устойчивый центральный градиент для предотвращения разрывов в нуле
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy(); xm = x.copy()
            xp[i] += eps; xm[i] -= eps
            g[i] = (self.f(xp) - self.f(xm)) / (2 * eps)
        return g

    def hess(self, x, eps=1e-5):
        # Численный расчет матрицы Гессе
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    xp = x.copy(); xm = x.copy()
                    xp[i] += eps; xm[i] -= eps
                    H[i, i] = (self.f(xp) - 2*self.f(x) + self.f(xm)) / (eps**2)
                else:
                    xpp = x.copy(); xpm = x.copy(); xmp = x.copy(); xmm = x.copy()
                    xpp[i] += eps; xpp[j] += eps
                    xpm[i] += eps; xpm[j] -= eps
                    xmp[i] -= eps; xmp[j] += eps
                    xmm[i] -= eps; xmm[j] -= eps
                    H[i, j] = (self.f(xpp) - self.f(xpm) - self.f(xmp) + self.f(xmm)) / (4 * eps**2)
        return H