import numpy as np


class AckleyFunction:
    name = "Ackley"
    x_opt = np.array([0.0, 0.0])

    def f(self, x):
        n = len(x)
        sum1 = np.sum(np.asarray(x)**2)
        sum2 = np.sum(np.cos(2.0 * np.pi * np.asarray(x)))
        return float(-20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
                     - np.exp(sum2 / n) + 20 + np.e)

    def grad(self, x, eps=1e-7):
        g = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            xp = np.array(x, dtype=float); xm = np.array(x, dtype=float)
            xp[i] += eps; xm[i] -= eps
            g[i] = (self.f(xp) - self.f(xm)) / (2 * eps)
        return g

    def hess(self, x, eps=1e-5):
        n = len(x)
        H = np.zeros((n, n))
        fx = self.f(x)
        for i in range(n):
            for j in range(n):
                if i == j:
                    xp = np.array(x, dtype=float); xm = np.array(x, dtype=float)
                    xp[i] += eps; xm[i] -= eps
                    H[i, i] = (self.f(xp) - 2 * fx + self.f(xm)) / (eps**2)
                else:
                    xpp = np.array(x, dtype=float); xpm = np.array(x, dtype=float)
                    xmp = np.array(x, dtype=float); xmm = np.array(x, dtype=float)
                    xpp[i] += eps; xpp[j] += eps
                    xpm[i] += eps; xpm[j] -= eps
                    xmp[i] -= eps; xmp[j] += eps
                    xmm[i] -= eps; xmm[j] -= eps
                    H[i, j] = (self.f(xpp) - self.f(xpm) - self.f(xmp) + self.f(xmm)) / (4 * eps**2)
        return H
