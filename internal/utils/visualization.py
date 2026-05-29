import numpy as np
import matplotlib.pyplot as plt


def plot_benchmark_graphs(df, metric="iters", group_by="n", fixed_val=10, title="", save_path=None):
    plt.figure(figsize=(10, 6))

    # Фильтруем данные в зависимости от того, по чему строим график
    if group_by == "n":
        sub_df = df[df["k"] == fixed_val]
        x_col = "n"
        xlabel = "Размерность пространства (n)"
    else:
        sub_df = df[df["n"] == fixed_val]
        x_col = "k"
        xlabel = "Число обусловленности (k)"

    for method in sub_df["method"].unique():
        method_df = sub_df[sub_df["method"] == method]
        plt.plot(method_df[x_col], method_df[metric], marker='o', label=method)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Итерации" if metric == "iters" else "Вызовы функции")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_contour_and_trajectories(func_obj, trajectories, title="", x_range=(-5, 5), y_range=(-5, 5), save_path=None):
    plt.figure(figsize=(8, 8))

    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)

    # Считаем значения функции на сетке
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func_obj.f(np.array([X[i, j], Y[i, j]]))

    plt.contour(X, Y, Z, levels=50, cmap="viridis", alpha=0.6)

    for name, path in trajectories.items():
        if len(path) > 0:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], marker='.', linestyle='-', linewidth=1.5, label=name)
            plt.plot(path[0, 0], path[0, 1], 'go')  # Старт
            plt.plot(path[-1, 0], path[-1, 1], 'ro')  # Финиш

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize="small")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
# from shortcuts import Axes3D  # Для старых версий, в новых достаточно projection='3d'


# ... (ваши старые функции plot_benchmark_graphs и plot_contour_and_trajectories остаются) ...

def plot_ackley_3d_side_view(func_obj, trajectories, title="Функция Экли в 3D (Вид сбоку)", x_range=(-3, 3),
                             y_range=(-3, 3), save_path=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Строим сетку для поверхности
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func_obj.f(np.array([X[i, j], Y[i, j]]))

    # Рисуем саму «зубчатую воронку» Экли полупрозрачной, чтобы видеть траектории
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.3)

    # Наносим траектории методов
    for name, path in trajectories.items():
        if len(path) > 0:
            path = np.array(path)
            # Считаем Z-координату (значение функции) для каждой точки пути
            z_path = [func_obj.f(p) for p in path]
            z_path = np.array(z_path)

            # Рисуем линию движения в 3D пространстве
            ax.plot(path[:, 0], path[:, 1], z_path, marker='.', linestyle='-', linewidth=2, label=name)
            # Отмечаем старт и финиш
            ax.scatter(path[0, 0], path[0, 1], z_path[0], color='green', s=50, zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], z_path[-1], color='red', s=50, zorder=5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('F(X1, X2) — Высота')

    # КРИТИЧЕСКИЙ МОМЕНТ: Меняем угол обзора, чтобы смотреть именно СБОКУ
    # elev — высота над плоскостью (установим пониже), azim — поворот вокруг оси Z
    ax.view_init(elev=30, azim=135)

    ax.legend(loc="best", fontsize="small")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()