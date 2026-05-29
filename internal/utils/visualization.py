import numpy as np
import matplotlib.pyplot as plt


def plot_contour_and_trajectories(f_obj, trajectories, title, x_range=(-3, 3), y_range=(-3, 3)):
    """
    Строит красивые линии уровня целевой функции и накладывает траектории методов.
    Использует адаптивный шаг и логарифмическую шкалу уровней для сложных функций.
    """
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)

    # Вычисление сетки значений Z
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f_obj.f(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))

    # Использование логарифмических уровней для овражных функций
    if "Rosenbrock" in f_obj.__class__.__name__ or "Ackley" in f_obj.__class__.__name__:
        levels = np.logspace(np.log10(Z.min() + 1e-3), np.log10(Z.max()), 40)
    else:
        levels = np.linspace(Z.min(), Z.max(), 40)

    contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.6)
    plt.colorbar(contour, label="f(x)")

    # Отрисовка траекторий
    markers = ['o', 's', '^', 'D', 'x', 'p', '*']
    for idx, (name, path) in enumerate(trajectories.items()):
        pts = np.array(path)
        if len(pts) == 0:
            continue
        plt.plot(pts[:, 0], pts[:, 1], marker=markers[idx % len(markers)],
                 label=name, markersize=4, linewidth=1.5, alpha=0.9)
        # Отмечаем старт и финиш
        plt.plot(pts[0, 0], pts[0, 1], 'go', markersize=8)
        plt.plot(pts[-1, 0], pts[-1, 1], 'ro', markersize=8)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_benchmark_graphs(df, metric, group_by="n", fixed_val=2, title=""):
    """
    Строит графики зависимостей вычислительных затрат от n или k.
    """
    plt.figure(figsize=(9, 6))
    if group_by == "n":
        sub_df = df[df['k'] == fixed_val]
        x_col = 'n'
        xlabel = "Размерность пространства (n)"
    else:
        sub_df = df[df['n'] == fixed_val]
        x_col = 'k'
        xlabel = "Число обусловленности (k)"

    for method in sub_df['method'].unique():
        m_df = sub_df[sub_df['method'] == method].sort_values(by=x_col)
        plt.plot(m_df[x_col], m_df[metric], marker='o', label=method, linewidth=2)

    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(metric, fontsize=11)
    if group_by == "k":
        plt.xscale('log')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()