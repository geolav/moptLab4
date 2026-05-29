import numpy as np
import pandas as pd

from internal.optimizers.cg import ConjugateGradientQuad, NonlinearCGFR, NonlinearCGPR
from internal.optimizers.newton import NewtonCholesky, NewtonDirectionSearch
from internal.optimizers.dogleg import PowellDogLeg
from internal.optimizers.quasi_newton import DFPOptimizer, BFGSOptimizer, LBFGSOptimizer
from internal.optimizers.scipy_newton import ScipyNewtonCG

from internal.functions.quadratic import QuadraticFunction
from internal.functions.rosenbrock import RosenbrockFunction
from internal.functions.himmelblau import HimmelblauFunction
from internal.functions.ackley import AckleyFunction

from internal.runner import ExperimentRunner
from internal.utils.visualization import plot_contour_and_trajectories, plot_benchmark_graphs


def get_all_optimizers():
    return {
        "CG-Quad": ConjugateGradientQuad(),
        "CG-FR": NonlinearCGFR(),
        "CG-PR": NonlinearCGPR(),
        "Newton-Cholesky": NewtonCholesky(),
        "Newton-Search": NewtonDirectionSearch(),
        "Powell-DogLeg": PowellDogLeg(),
        "DFP": DFPOptimizer(),
        "BFGS": BFGSOptimizer(),
        "L-BFGS": LBFGSOptimizer(memory_size=5),
        "SciPy-Newton-CG": ScipyNewtonCG()
    }


def run_all_tasks():
    opts = get_all_optimizers()
    runner = ExperimentRunner(opts)

    # Настройки Pandas для красивого вывода таблиц в консоль без обрезания строк
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}' if not x.is_integer() else f'{int(x)}')

    print("=== Выполнение Задачи 1: Исследование на генераторе квадратичных функций ===")
    dims = [2, 10, 50, 100]
    conds = [1, 10, 100, 1000]
    df_res = runner.run_quadratic_suite(dims, conds, repeats=2)

    # Вывод красивой таблицы стандартными средствами pandas
    print("\nРезультаты бенчмарка квадратичных функций:")
    print("-" * 80)
    print(df_res.to_string(index=False))
    print("-" * 80)

    # Построение графиков
    plot_benchmark_graphs(df_res, metric="iters", group_by="n", fixed_val=10,
                          title="Зависимость числа итераций от n при фиксированном k=10")
    plot_benchmark_graphs(df_res, metric="iters", group_by="k", fixed_val=50,
                          title="Зависимость числа итераций от числа обусловленности k при n=50")

    print("\n=== Выполнение Задачи 2: Траектории на 2D квадратичной функции ===")
    quad_2d = QuadraticFunction(n=2, k=10)
    start_points = [
        np.array([-2.0, 2.0]),
        np.array([2.5, 2.5]),
        np.array([-2.5, -1.0]),
        np.array([0.1, -2.5]),
        np.array([3.0, -2.0])
    ]

    for idx, p_start in enumerate(start_points):
        trajectories = {}
        for name, opt in opts.items():
            from internal.utils.counter import ProfilerOracle
            oracle = ProfilerOracle(quad_2d)
            res = opt.minimize(oracle, p_start)
            trajectories[name] = res.path

        plot_contour_and_trajectories(quad_2d, trajectories,
                                      title=f"2D Квадратичная функция (k=10). Старт из точки {p_start}",
                                      x_range=(-4, 4), y_range=(-4, 4))

    print("\n=== Выполнение Задачи 3: Исследование сложных нелинейных функций ===")
    # 3.1 Розенброк
    rosen = RosenbrockFunction()
    r_starts = [np.array([-1.2, 1.0]), np.array([2.0, -1.5])]
    trajectories_r = {}
    for name, opt in opts.items():
        from internal.utils.counter import ProfilerOracle
        oracle = ProfilerOracle(rosen)
        res = opt.minimize(oracle, r_starts[0])
        trajectories_r[name] = res.path
    plot_contour_and_trajectories(rosen, trajectories_r, title="Траектории на функции Розенброка", x_range=(-2, 3),
                                  y_range=(-1, 3))

    # 3.2 Химмельблау
    himmel = HimmelblauFunction()
    h_starts = [np.array([0.0, 0.0]), np.array([-2.0, -2.0])]
    trajectories_h = {}
    for name, opt in opts.items():
        from internal.utils.counter import ProfilerOracle
        oracle = ProfilerOracle(himmel)
        res = opt.minimize(oracle, h_starts[0])
        trajectories_h[name] = res.path
    plot_contour_and_trajectories(himmel, trajectories_h, title="Траектории на функции Химмельблау", x_range=(-5, 5),
                                  y_range=(-5, 5))

    # 3.3 Экли
    ackley = AckleyFunction()
    trajectories_a = {}