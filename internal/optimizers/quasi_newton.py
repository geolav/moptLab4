import numpy as np
from typing import Callable
from ..utils import OptResult, CallCounter
from .cg import wolfe_line_search, MAX_ITER


def dfp(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    n = len(x)
    I = np.eye(n)
    H = I.copy()
    traj = [x.copy()]
    g = cg(x)
    fx = cf(x)

    for k in range(max_iter):
        if np.linalg.norm(g) < eps:
            return OptResult(x, fx, k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        p = -H @ g
        if float(p @ g) >= 0:
            H = I.copy()
            p = -g
        alpha = wolfe_line_search(cf, cg, x, p, fx, g)
        x_next = x + alpha * p
        traj.append(x_next.copy())
        f_next = cf(x_next)
        g_next = cg(x_next)
        if abs(f_next - fx) <= 1e-14 * (1.0 + abs(fx)):
            conv = np.linalg.norm(g_next) < eps
            status = "Сходимость по градиенту" if conv else "Стагнация по функции"
            return OptResult(x_next, f_next, k + 1, cf.count, cg.count, ch.count,
                             conv, status, traj)
        s = x_next - x
        y = g_next - g
        sy = float(s @ y)
        if sy > 1e-10:
            Hy = H @ y
            H = H + np.outer(s, s) / sy - np.outer(Hy, Hy) / float(y @ Hy)
        x, g = x_next, g_next
        fx = f_next

    return OptResult(x, fx, max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)


def bfgs(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    n = len(x)
    I = np.eye(n)
    H = I.copy()
    traj = [x.copy()]
    g = cg(x)
    fx = cf(x)

    for k in range(max_iter):
        if np.linalg.norm(g) < eps:
            return OptResult(x, fx, k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        p = -H @ g
        if float(p @ g) >= 0:
            H = I.copy()
            p = -g
        alpha = wolfe_line_search(cf, cg, x, p, fx, g)
        x_next = x + alpha * p
        traj.append(x_next.copy())
        f_next = cf(x_next)
        g_next = cg(x_next)
        if abs(f_next - fx) <= 1e-14 * (1.0 + abs(fx)):
            conv = np.linalg.norm(g_next) < eps
            status = "Сходимость по градиенту" if conv else "Стагнация по функции"
            return OptResult(x_next, f_next, k + 1, cf.count, cg.count, ch.count,
                             conv, status, traj)
        s = x_next - x
        y = g_next - g
        sy = float(y @ s)
        if sy > 1e-10:
            rho = 1.0 / sy
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        x, g = x_next, g_next
        fx = f_next

    return OptResult(x, fx, max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)


def lbfgs(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER,
    memory_size: int = 5
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    traj = [x.copy()]
    history = []
    g = cg(x)
    fx = cf(x)

    for k in range(max_iter):
        if np.linalg.norm(g) < eps:
            return OptResult(x, fx, k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)

        q = g.copy()
        alphas = []
        for s_i, y_i, rho_i in reversed(history):
            a_i = rho_i * float(s_i @ q)
            alphas.append(a_i)
            q = q - a_i * y_i

        if history:
            s_l, y_l, _ = history[-1]
            gamma = float(s_l @ y_l) / float(y_l @ y_l)
        else:
            gamma = 1.0

        r = gamma * q
        for (s_i, y_i, rho_i), a_i in zip(history, reversed(alphas)):
            b_i = rho_i * float(y_i @ r)
            r = r + s_i * (a_i - b_i)

        p = -r
        alpha = wolfe_line_search(cf, cg, x, p, fx, g)
        x_next = x + alpha * p
        traj.append(x_next.copy())
        f_next = cf(x_next)
        g_next = cg(x_next)
        if abs(f_next - fx) <= 1e-14 * (1.0 + abs(fx)):
            conv = np.linalg.norm(g_next) < eps
            status = "Сходимость по градиенту" if conv else "Стагнация по функции"
            return OptResult(x_next, f_next, k + 1, cf.count, cg.count, ch.count,
                             conv, status, traj)
        s = x_next - x
        y = g_next - g
        sy = float(y @ s)
        if sy > 1e-10:
            history.append((s, y, 1.0 / sy))
            if len(history) > memory_size:
                history.pop(0)
        x, g = x_next, g_next
        fx = f_next

    return OptResult(x, fx, max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)
