import numpy as np
from typing import Callable
from ..utils import OptResult, CallCounter

MAX_ITER = 5000


def armijo_line_search(cf, x, p, fx, gp, alpha0=1.0, c1=1e-4, q=0.5, max_back=50):
    alpha = alpha0
    for _ in range(max_back):
        if cf(x + alpha * p) <= fx + c1 * alpha * gp:
            return alpha
        alpha *= q
    return alpha


def wolfe_line_search(cf, cg, x, p, f0, g0, c1=1e-4, c2=0.9,
                      alpha0=1.0, alpha_max=1e10, max_iter=30):
    dphi0 = float(g0 @ p)
    if dphi0 >= 0:
        return 0.0
    phi0 = f0

    def phi(a):
        return cf(x + a * p)

    def dphi(a):
        return float(cg(x + a * p) @ p)

    def zoom(a_lo, a_hi, phi_lo):
        for _ in range(max_iter):
            a = 0.5 * (a_lo + a_hi)
            phi_a = phi(a)
            if phi_a > phi0 + c1 * a * dphi0 or phi_a >= phi_lo:
                a_hi = a
            else:
                dphi_a = dphi(a)
                if abs(dphi_a) <= -c2 * dphi0:
                    return a
                if dphi_a * (a_hi - a_lo) >= 0:
                    a_hi = a_lo
                a_lo = a
                phi_lo = phi_a
        return 0.5 * (a_lo + a_hi)

    a_prev = 0.0
    phi_prev = phi0
    a = alpha0
    for i in range(max_iter):
        phi_a = phi(a)
        if phi_a > phi0 + c1 * a * dphi0 or (i > 0 and phi_a >= phi_prev):
            return zoom(a_prev, a, phi_prev)
        dphi_a = dphi(a)
        if abs(dphi_a) <= -c2 * dphi0:
            return a
        if dphi_a >= 0:
            return zoom(a, a_prev, phi_a)
        a_prev = a
        phi_prev = phi_a
        a = min(2.0 * a, alpha_max)
    return a


def cg_quadratic(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    traj = [x.copy()]
    A = ch(x)
    g = cg(x)
    p = -g

    for k in range(max_iter):
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        Ap = A @ p
        pAp = float(p @ Ap)
        if pAp <= 0:
            return OptResult(x, cf(x), k, cf.count, cg.count, ch.count,
                             False, "Направление неположительной кривизны", traj)
        alpha = float(g @ g) / pAp
        x = x + alpha * p
        traj.append(x.copy())
        g_next = g + alpha * Ap
        beta = float(g_next @ g_next) / float(g @ g)
        p = -g_next + beta * p
        g = g_next

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)


def nonlinear_cg_fr(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    traj = [x.copy()]
    g = cg(x)
    p = -g
    g_sq = float(g @ g)
    fx = cf(x)

    for k in range(max_iter):
        if np.sqrt(g_sq) < eps:
            return OptResult(x, fx, k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        alpha = wolfe_line_search(cf, cg, x, p, fx, g, c2=0.1)
        x = x + alpha * p
        traj.append(x.copy())
        f_next = cf(x)
        g = cg(x)
        g_sq_next = float(g @ g)
        if abs(f_next - fx) <= 1e-14 * (1.0 + abs(fx)):
            conv = np.sqrt(g_sq_next) < eps
            status = "Сходимость по градиенту" if conv else "Стагнация по функции"
            return OptResult(x, f_next, k + 1, cf.count, cg.count, ch.count,
                             conv, status, traj)
        beta = g_sq_next / g_sq
        p = -g + beta * p
        if float(g @ p) >= 0:
            p = -g
        g_sq = g_sq_next
        fx = f_next

    return OptResult(x, fx, max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)


def nonlinear_cg_pr(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    traj = [x.copy()]
    g = cg(x)
    p = -g
    g_sq = float(g @ g)
    fx = cf(x)

    for k in range(max_iter):
        if np.sqrt(g_sq) < eps:
            return OptResult(x, fx, k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        alpha = wolfe_line_search(cf, cg, x, p, fx, g, c2=0.1)
        x = x + alpha * p
        traj.append(x.copy())
        f_next = cf(x)
        g_new = cg(x)
        if abs(f_next - fx) <= 1e-14 * (1.0 + abs(fx)):
            conv = np.linalg.norm(g_new) < eps
            status = "Сходимость по градиенту" if conv else "Стагнация по функции"
            return OptResult(x, f_next, k + 1, cf.count, cg.count, ch.count,
                             conv, status, traj)
        y = g_new - g
        beta = max(0.0, float(g_new @ y) / g_sq)
        p = -g_new + beta * p
        if float(g_new @ p) >= 0:
            p = -g_new
        g = g_new
        g_sq = float(g @ g)
        fx = f_next

    return OptResult(x, fx, max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)
