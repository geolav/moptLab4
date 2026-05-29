import numpy as np

class ProfilerOracle:
    """
    Обертка над функцией для автоматического подсчета количества вычислений
    значения функции, ее градиента и матрицы Гессе.
    """
    def __init__(self, target_function):
        self.target = target_function
        self.f_count = 0
        self.g_count = 0
        self.h_count = 0

    def f(self, x):
        self.f_count += 1
        return self.target.f(x)

    def grad(self, x):
        self.g_count += 1
        return self.target.grad(x)

    def hess(self, x):
        self.h_count += 1
        return self.target.hess(x)

    def reset(self):
        self.f_count = 0
        self.g_count = 0
        self.h_count = 0