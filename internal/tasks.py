import numpy as np

from internal.functions import (
    QuadraticFunction,
    RosenbrockFunction,
    HimmelblauFunction,
    AckleyFunction,
)
from internal.optimizers import (
    cg_quadratic,
    nonlinear_cg_fr,
    nonlinear_cg_pr,
    newton_cholesky,
    newton_search,
    powell_dogleg,
    dfp,
    bfgs,
    lbfgs,
    scipy_newton_cg,
)
from internal.utils.visualization import (
    plot_contour_with_trajectory,
    plot_iter_vs_step,
    plot_metric_vs_param,
    print_table,
)


OPTIMIZERS = {
    "CG (квадратичный)":  cg_quadratic,
    "CG-FR":              nonlinear_cg_fr,
    "CG-PR":              nonlinear_cg_pr,
    "Ньютон (Холецкий)":  newton_cholesky,
    "Ньютон (поиск)":     newton_search,
    "Dog Leg":            powell_dogleg,
    "DFP":                dfp,
    "BFGS":               bfgs,
    "L-BFGS":             lbfgs,
    "Ньютон-CG (scipy)":  scipy_newton_cg,
}

SHORT = {
    "CG (квадратичный)":  "CGq",
    "CG-FR":              "FR",
    "CG-PR":              "PR",
    "Ньютон (Холецкий)":  "NCh",
    "Ньютон (поиск)":     "NLs",
    "Dog Leg":            "DL",
    "DFP":                "DFP",
    "BFGS":               "BFGS",
    "L-BFGS":             "LBFGS",
    "Ньютон-CG (scipy)":  "sNCG",
}


def _run(opt, func, x0, eps=1e-8):
    return opt(func.f, func.grad, func.hess, np.array(x0, dtype=float), eps=eps)


def quadratic_benchmark(eps: float = 1e-8):
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 1: Генератор квадратичных функций")
    print("=" * 70)

    dims = [2, 10, 50, 100]
    conds = [1, 10, 100, 1000]
    repeats = 2

    iters_mean = {}
    for n in dims:
        for k in conds:
            for name, opt in OPTIMIZERS.items():
                vals = []
                for rep in range(repeats):
                    func = QuadraticFunction(n, k, seed=42 + rep)
                    r = _run(opt, func, np.ones(n) * 2.5, eps)
                    vals.append(r.n_iter)
                iters_mean[(n, k, name)] = float(np.mean(vals))

    header = ["n", "k"] + [SHORT[name] for name in OPTIMIZERS]
    rows = []
    for n in dims:
        for k in conds:
            row = [n, k] + [f"{iters_mean[(n, k, name)]:.0f}" for name in OPTIMIZERS]
            rows.append(row)
    print_table(header, rows,
                title="Бенчмарк: среднее число итераций",
                slug="bench_iters")

    series_n = {SHORT[name]: [iters_mean[(n, 10, name)] for n in dims]
                for name in OPTIMIZERS}
    plot_metric_vs_param(dims, series_n,
                         "Число итераций от размерности n (k=10)",
                         "Размерность n", filename="fig_bench_dim.png", logy=True)

    series_k = {SHORT[name]: [iters_mean[(50, k, name)] for k in conds]
                for name in OPTIMIZERS}
    plot_metric_vs_param(conds, series_k,
                         "Число итераций от обусловленности k (n=50)",
                         "Число обусловленности k", filename="fig_bench_cond.png",
                         logx=True, logy=True)


def quadratic_trajectories_2d(eps: float = 1e-8):
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2: Траектории на 2D квадратичной функции")
    print("=" * 70)

    func = QuadraticFunction(n=2, k=10, seed=7)
    func.x_opt = np.zeros(2)
    func.b = np.zeros(2)

    starts = [
        np.array([-2.0, 2.0]), np.array([2.5, 2.5]),
        np.array([-2.5, -1.0]), np.array([0.1, -2.5]),
        np.array([3.0, -2.0]),
    ]

    agg = {name: {"it": [], "f": [], "g": [], "h": [], "ok": 0} for name in OPTIMIZERS}

    for idx, x0 in enumerate(starts):
        trajs = {}
        for name, opt in OPTIMIZERS.items():
            r = _run(opt, func, x0, eps)
            trajs[name] = r.trajectory
            agg[name]["it"].append(r.n_iter)
            agg[name]["f"].append(r.n_f)
            agg[name]["g"].append(r.n_grad)
            agg[name]["h"].append(r.n_hess)
            agg[name]["ok"] += int(r.converged)
        plot_contour_with_trajectory(
            func.f,
            f"2D квадратичная (k=10). Старт {idx + 1}: {np.round(x0, 1)}",
            trajs, xlim=(-4, 4), ylim=(-4, 4),
            filename=f"fig_quad2d_start_{idx + 1}.png",
        )

    rows = []
    for name in OPTIMIZERS:
        a = agg[name]
        rows.append([
            name,
            f"{np.mean(a['it']):.0f}",
            f"{np.mean(a['f']):.0f}",
            f"{np.mean(a['g']):.0f}",
            f"{np.mean(a['h']):.0f}",
            f"{a['ok']}/{len(starts)}",
        ])
    print_table(
        ["Метод", "Итерации", "f_calls", "grad_calls", "hess_calls", "Сошлось"],
        rows, title="2D квадратичная: сводка по 5 стартам", slug="quad2d_summary")


def _complex_one(func, x0, xlim, ylim, slug, eps=1e-8):
    rows = []
    trajs = {}
    for name, opt in OPTIMIZERS.items():
        r = _run(opt, func, x0, eps)
        trajs[name] = r.trajectory
        rows.append([
            name,
            r.n_iter if r.converged else "—",
            r.n_f, r.n_grad, r.n_hess,
            f"{r.f_val:.4e}",
            "✓" if r.converged else "✗",
        ])
    print_table(
        ["Метод", "Итерации", "f_calls", "grad_calls", "hess_calls", "f(x*)", "Сошлось"],
        rows, title=f"{func.name}: старт {np.round(x0, 2)}", slug=slug)
    plot_contour_with_trajectory(
        func.f, f"Траектории на функции {func.name}",
        trajs, xlim, ylim, filename=f"fig_{slug}.png")


def complex_functions(eps: float = 1e-8):
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 3: Сложные нелинейные функции")
    print("=" * 70)

    _complex_one(RosenbrockFunction(), np.array([-1.2, 1.0]),
                 (-2, 3), (-1, 3), "rosenbrock", eps)
    _complex_one(HimmelblauFunction(), np.array([-2.0, 2.0]),
                 (-5, 5), (-5, 5), "himmelblau", eps)
    _complex_one(AckleyFunction(), np.array([1.5, 1.5]),
                 (-3, 3), (-3, 3), "ackley", eps)


def lbfgs_memory_study(eps: float = 1e-8):
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 4: Влияние размера памяти m в L-BFGS")
    print("=" * 70)

    func = QuadraticFunction(n=50, k=100, seed=42)
    x0 = np.ones(50) * 3.0

    rows = []
    ms = [1, 2, 5, 10, 20, 50]
    iters = []
    for m in ms:
        r = lbfgs(func.f, func.grad, func.hess, x0, eps=eps, memory_size=m)
        iters.append(r.n_iter)
        rows.append([
            m, r.n_iter, r.n_f, r.n_grad,
            "✓" if r.converged else "✗",
        ])
    print_table(
        ["m", "Итерации", "f_calls", "grad_calls", "Сошлось"],
        rows, title="Влияние памяти m в L-BFGS (n=50, k=100)", slug="lbfgs_memory")

    plot_iter_vs_step(ms, iters,
                      "Число итераций от размера памяти m (L-BFGS)",
                      xlabel="Размер памяти m", filename="fig_lbfgs_memory.png")
