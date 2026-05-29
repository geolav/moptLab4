import os
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

from internal.utils.visualization import plot_contour_and_trajectories, plot_benchmark_graphs, plot_ackley_3d_side_view

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


def run_and_log_experiment(func_obj, start_points, exp_name, opts):
    """
    Вспомогательная функция для прогона экспериментов,
    сохранения детальной таблицы логов и возврата траекторий.
    """
    records = []
    trajectories_by_point = []

    for idx, p_start in enumerate(start_points):
        trajectories = {}
        for name, opt in opts.items():
            from internal.utils.counter import ProfilerOracle
            oracle = ProfilerOracle(func_obj)
            try:
                res = opt.minimize(oracle, p_start.copy())
                trajectories[name] = res.path
                records.append({
                    "Task": exp_name,
                    "Point": f"Pt_{idx + 1}",
                    "Method": name,
                    "Status": res.status,
                    "Iters": res.iters,
                    "F_evals": res.f_evals,
                    "G_evals": res.g_evals,
                    "H_evals": res.h_evals
                })
            except Exception as e:
                records.append({
                    "Task": exp_name, "Point": f"Pt_{idx + 1}", "Method": name,
                    "Status": f"Failed", "Iters": np.nan, "F_evals": np.nan,
                    "G_evals": np.nan, "H_evals": np.nan
                })
        trajectories_by_point.append(trajectories)

    df = pd.DataFrame(records)
    print(f"\nДетали эксперимента: {exp_name}")
    print("-" * 85)
    print(df.to_string(index=False))
    print("-" * 85)
    df.to_csv(f"results/tables/{exp_name}_details.csv", index=False, encoding="utf-8")

    return trajectories_by_point


def run_all_tasks():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    opts = get_all_optimizers()
    runner = ExperimentRunner(opts)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: f'{x:.0f}' if pd.notnull(x) else 'NaN')

    print("=== Выполнение Задачи 1: Исследование на генераторе квадратичных функций ===")
    dims = [2, 10, 50, 100]
    conds = [1, 10, 100, 1000]
    df_res = runner.run_quadratic_suite(dims, conds, repeats=2)

    print("\nРезультаты бенчмарка квадратичных функций:")
    print("-" * 80)
    print(df_res.to_string(index=False))
    print("-" * 80)
    df_res.to_csv("results/tables/quadratic_benchmark.csv", index=False, encoding="utf-8")

    plot_benchmark_graphs(df_res, metric="iters", group_by="n", fixed_val=10,
                          title="Зависимость числа итераций от n при фиксированном k=10",
                          save_path="results/plots/benchmark_by_dimension.png")
    plot_benchmark_graphs(df_res, metric="iters", group_by="k", fixed_val=50,
                          title="Зависимость числа итераций от числа обусловленности k при n=50",
                          save_path="results/plots/benchmark_by_condition.png")

    print("\n=== Выполнение Задачи 2: Траектории на 2D квадратичной функции ===")
    quad_2d = QuadraticFunction(n=2, k=10)
    start_points_2d = [
        np.array([-2.0, 2.0]), np.array([2.5, 2.5]),
        np.array([-2.5, -1.0]), np.array([0.1, -2.5]), np.array([3.0, -2.0])
    ]
    traj_2d = run_and_log_experiment(quad_2d, start_points_2d, "Quadratic_2D", opts)

    for idx, trajectories in enumerate(traj_2d):
        plot_contour_and_trajectories(quad_2d, trajectories,
                                      title=f"2D Квадратичная функция (k=10). Старт {idx + 1}",
                                      x_range=(-4, 4), y_range=(-4, 4),
                                      save_path=f"results/plots/quad_2d_trajectory_point_{idx + 1}.png")

    print("\n=== Выполнение Задачи 3: Исследование сложных нелинейных функций ===")

    # 3.1 Розенброк
    rosen = RosenbrockFunction()
    r_starts = [np.array([-1.2, 1.0])]
    traj_r = run_and_log_experiment(rosen, r_starts, "Rosenbrock", opts)
    plot_contour_and_trajectories(rosen, traj_r[0], title="Траектории на функции Розенброка",
                                  x_range=(-2, 3), y_range=(-1, 3),
                                  save_path="results/plots/rosenbrock_trajectory.png")

    # 3.2 Химмельблау
    himmel = HimmelblauFunction()
    h_starts = [np.array([0.0, 0.0])]
    traj_h = run_and_log_experiment(himmel, h_starts, "Himmelblau", opts)
    plot_contour_and_trajectories(himmel, traj_h[0], title="Траектории на функции Химмельблау",
                                  x_range=(-5, 5), y_range=(-5, 5),
                                  save_path="results/plots/himmelblau_trajectory.png")

    # 3.3 Экли
    ackley = AckleyFunction()
    a_starts = [np.array([1.5, 1.5])]
    traj_a = run_and_log_experiment(ackley, a_starts, "Ackley", opts)

    # Старый 2D график сверху
    plot_contour_and_trajectories(ackley, traj_a[0], title="Траектории на функции Экли (Вид сверху)",
                                  x_range=(-3, 3), y_range=(-3, 3),
                                  save_path="results/plots/ackley_trajectory_2d.png")

    # НАШ НОВЫЙ 3D ГРАФИК СБОКУ
    plot_ackley_3d_side_view(ackley, traj_a[0], title="Траектории на функции Экли (Вид сбоку в 3D)",
                             x_range=(-3, 3), y_range=(-3, 3),
                             save_path="results/plots/ackley_trajectory_3d_side.png")

    print("\n=== Выполнение Задачи 4: Исследование влияния размера памяти L-BFGS ===")
    records_lbfgs = []
    for m_size in [2, 5, 10, 20]:
        lbfgs_opt = LBFGSOptimizer(memory_size=m_size)
        func = QuadraticFunction(n=50, k=100, seed=42)
        from internal.utils.counter import ProfilerOracle
        oracle = ProfilerOracle(func)
        res = lbfgs_opt.minimize(oracle, np.ones(50) * 3.0)
        records_lbfgs.append({"m_size": m_size, "iters": res.iters, "f_evals": res.f_evals})

    df_lbfgs = pd.DataFrame(records_lbfgs)
    print("\nВлияние размера памяти m на сходимость L-BFGS:")
    print("-" * 50)
    print(df_lbfgs.to_string(index=False))
    print("-" * 50)
    df_lbfgs.to_csv("results/tables/lbfgs_memory_study.csv", index=False, encoding="utf-8")