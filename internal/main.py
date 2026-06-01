from internal.tasks import (
    quadratic_benchmark,
    quadratic_trajectories_2d,
    complex_functions,
    lbfgs_memory_study,
)
from internal.runner import run


def main():
    run([
        quadratic_benchmark,
        quadratic_trajectories_2d,
        complex_functions,
        lbfgs_memory_study,
    ], output_root="results", save_graphs=True, save_tables=True)


if __name__ == "__main__":
    main()
