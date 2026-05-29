import numpy as np


class QuadraticFunction:
    """
    Генератор выпуклых квадратичных функций с заданным числом обусловленности k и размерностью n.
    """

    def __init__(self, n, k, seed=42):
        np.random.seed(seed)

        # 1. Генерация собственных значений от 1 до k
        if n > 1:
            eigenvalues = np.linspace(1, k, n)
            eigenvalues[0] = 1
            eigenvalues[-1] = k
        else:
            eigenvalues = np.array([1.0])

        # 2. Случайная ортогональная матрица через QR разложение
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)

        # 3. Сборка матрицы A
        self.A = Q @ np.diag(eigenvalues) @ Q.T

        # 4. Выбор точки минимума и вычисление вектора b
        self.x_star = np.random.randn(n)
        self.b = self.A @ self.x_star
        self.c = 0.0

    def f(self, x):
        return 0.5 * np.dot(x, self.A @ x) - np.dot(self.b, x) + self.c

    def grad(self, x):
        return self.A @ x - self.b

    def hess(self, x):
        return self.A