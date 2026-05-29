import numpy as np
import pandas as pd
from internal.utils.counter import ProfilerOracle


class ExperimentRunner:
    """
    Класс для автоматизации пакетных запусков наборов оптимизаторов на заданных целях.
    """

    def __init__(self, optimizers_dict):
        self.optimizers = optimizers_dict

    def run_quadratic_suite(self, dimensions, conditions, repeats=3):
        records = []
        for n in dimensions:
            for k in conditions:
                for rep in range(repeats):
                    from internal.functions.quadratic import QuadraticFunction
                    func = QuadraticFunction(n, k, seed=42 + rep)

                    for name, opt in self.optimizers.items():
                        oracle = ProfilerOracle(func)
                        x0 = np.ones(n) * 2.5

                        try:
                            res = opt.minimize(oracle, x0, tol=1e-8)
                            records.append({
                                "n": n, "k": k, "method": name,
                                "iters": res.iters, "f_evals": res.f_evals,
                                "g_evals": res.g_evals, "h_evals": res.h_evals,
                                "status": res.status
                            })
                        except Exception as e:
                            # В случае падения метода пишем лог, чтобы не ломать весь бенчмарк
                            records.append({
                                "n": n, "k": k, "method": name,
                                "iters": np.nan, "f_evals": np.nan,
                                "g_evals": np.nan, "h_evals": np.nan,
                                "status": f"Failed: {str(e)}"
                            })

        df = pd.DataFrame(records)

        # Исправлено: явно указываем numeric_only=True, чтобы pandas не пытался брать среднее от строк статуса
        summary = df.groupby(['n', 'k', 'method']).mean(numeric_only=True).reset_index()
        return summary

    def run_fixed_function(self, func_obj, x0_points):
        results = {}
        for name, opt in self.optimizers.items():
            results[name] = {}
            for idx, x0 in enumerate(x0_points):
                oracle = ProfilerOracle(func_obj)
                res = opt.minimize(oracle, x0, tol=1e-8)
                results[name][f"Point_{idx}"] = res
        return results