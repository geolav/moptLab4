import numpy as np
from typing import Callable
from scipy.linalg import cho_factor, cho_solve
from ..utils import OptResult, CallCounter
from .cg import armijo_line_search, MAX_ITER


def newton_cholesky(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        H = ch(x)
        try:
            c, low = cho_factor(H)
            p = cho_solve((c, low), -g)
        except np.linalg.LinAlgError:
            return OptResult(x, cf(x), k, cf.count, cg.count, ch.count,
                             False, "Гессиан не положительно определён", traj)
        x = x + p
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)


def newton_search(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    n = len(x)
    I = np.eye(n)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        H = ch(x)
        w = np.linalg.eigvalsh(H)
        if w.min() <= 1e-4:
            H = H + (1e-3 - w.min()) * I
        p = np.linalg.solve(H, -g)
        fx = cf(x)
        alpha = armijo_line_search(cf, x, p, fx, float(g @ p))
        x = x + alpha * p
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)
