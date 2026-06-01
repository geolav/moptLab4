import numpy as np


class QuadraticFunction:
    def __init__(self, n, k, seed=42):
        np.random.seed(seed)

        if n > 1:
            eigenvalues = np.linspace(1.0, float(k), n)
            eigenvalues[0] = 1.0
            eigenvalues[-1] = float(k)
        else:
            eigenvalues = np.array([1.0])

        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)

        self.A = Q @ np.diag(eigenvalues) @ Q.T
        self.x_opt = np.random.randn(n)
        self.b = self.A @ self.x_opt
        self.c = 0.0
        self.name = f"Quadratic (n={n}, k={k})"

    def f(self, x):
        return float(0.5 * x @ self.A @ x - self.b @ x + self.c)

    def grad(self, x):
        return self.A @ x - self.b

    def hess(self, x):
        return self.A
