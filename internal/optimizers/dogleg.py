import numpy as np
from typing import Callable
from ..utils import OptResult, CallCounter
from .cg import MAX_ITER


def powell_dogleg(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER,
    max_tr: float = 10.0, initial_tr: float = 1.0
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    x = x0.copy().astype(float)
    n = len(x)
    I = np.eye(n)
    tr = initial_tr
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        ng = np.linalg.norm(g)
        if ng < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, ch.count,
                             True, "Сходимость по градиенту", traj)
        if tr < 1e-12:
            return OptResult(x, cf(x), k, cf.count, cg.count, ch.count,
                             False, "Стагнация: малый радиус доверия", traj)

        B = ch(x)
        w = np.linalg.eigvalsh(B)
        if w.min() <= 1e-4:
            B = B + (1e-3 - w.min()) * I

        gBg = float(g @ (B @ g))
        tau = 1.0 if gBg <= 0 else min(1.0, ng**3 / (tr * gBg))
        pC = -tau * (tr / ng) * g
        npC = np.linalg.norm(pC)

        try:
            pN = np.linalg.solve(B, -g)
        except np.linalg.LinAlgError:
            pN = pC
        npN = np.linalg.norm(pN)

        if npN <= tr:
            p = pN
        elif npC >= tr:
            p = (tr / npC) * pC if npC > 1e-12 else np.zeros_like(pC)
        else:
            d = pN - pC
            a = float(d @ d)
            b = 2 * float(pC @ d)
            c = float(pC @ pC) - tr**2
            disc = b * b - 4 * a * c
            if disc < 0 or a == 0:
                p = pC
            else:
                t = (-b + np.sqrt(disc)) / (2 * a)
                p = pC + t * d

        mp = float(g @ p) + 0.5 * float(p @ (B @ p))
        fx = cf(x)
        f_new = cf(x + p)
        pred = -mp
        rho = 0.0 if pred < 1e-12 else (fx - f_new) / pred

        if rho < 0.25:
            tr *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(p) - tr) < 1e-8:
            tr = min(2.0 * tr, max_tr)

        if rho > 0:
            x = x + p
            traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, ch.count,
                     False, "Достигнут лимит итераций", traj)
