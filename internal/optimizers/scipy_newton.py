import numpy as np
from typing import Callable
from scipy.optimize import minimize
from ..utils import OptResult, CallCounter
from .cg import MAX_ITER


def scipy_newton_cg(
    f: Callable, grad: Callable, hess: Callable,
    x0: np.ndarray, eps: float = 1e-8, max_iter: int = MAX_ITER
) -> OptResult:
    cf = CallCounter(f); cg = CallCounter(grad); ch = CallCounter(hess)

    traj = [np.array(x0, dtype=float)]

    def cb(xk):
        traj.append(np.array(xk, dtype=float))

    res = minimize(
        fun=cf, x0=x0.copy().astype(float), method='Newton-CG',
        jac=cg, hess=ch, tol=eps, callback=cb,
        options={'maxiter': max_iter}
    )

    return OptResult(res.x, float(res.fun), int(res.nit),
                     cf.count, cg.count, ch.count,
                     bool(res.success), str(res.message), traj)
