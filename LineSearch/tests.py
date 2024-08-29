from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from functions import Function, QuadraticFunction, Rosenbrock2D
from line_search import SearchType, gradient_descent, newtons_method_line_search

GRID_LOWER_X = -2
GRID_LOWER_Y = -2
GRID_UPPER_X = 2
GRID_UPPER_Y = 2
GRID_SAMPLE_DENSITY = 100
GRID_CONTOURS_LEVELS = 100


def plot_line_search(func: Function, descent_history: List, title: str, display_log_norm_contours=False):
    descent_history = np.array(descent_history)
    x = np.linspace(GRID_LOWER_X, GRID_UPPER_X, GRID_SAMPLE_DENSITY)
    y = np.linspace(GRID_LOWER_Y, GRID_UPPER_Y, GRID_SAMPLE_DENSITY)
    X, Y = np.meshgrid(x, y)
    vals = np.vectorize(lambda xi, yi: func.f(np.array([xi, yi])))(X, Y)

    plt.figure(figsize=(10, 8))
    norm = matplotlib.colors.Normalize()
    if display_log_norm_contours:
        norm = matplotlib.colors.LogNorm()

    cp = plt.contourf(X, Y, vals, cmap='viridis', levels=GRID_CONTOURS_LEVELS, norm=norm)
    plt.colorbar(cp, label='Contour levels')

    plt.plot(descent_history[:, 0], descent_history[:, 1], 'ro-', markersize=5, label='Gradient Descent Path')
    plt.scatter(descent_history[0, 0], descent_history[0, 1], color='red', marker='o', s=100,
                zorder=5, label='Start')
    plt.scatter(descent_history[-1, 0], descent_history[-1, 1], color='blue', marker='x', s=100,
                zorder=5, label='End')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def test_quadratic_form(Q: np.ndarray, search_type: SearchType, starting_point: np.ndarray):
    search_type_str = "exact" if (search_type == SearchType.Exact) else "inexact"
    Q_str = np.array2string(Q, separator=', ', formatter={'all': lambda x: f'{x:.1f}'})
    Q_str = Q_str.replace('\n', ' ')

    func = QuadraticFunction(Q)

    x, history = gradient_descent(func, starting_point, search_type)
    plot_line_search(func, history,
                     f"Quadratic Form gradient descent, {search_type_str}, Q: {Q_str}")
    x, history = newtons_method_line_search(func, starting_point, search_type)
    plot_line_search(func, history,
                     f"Quadratic Form Newtons method, {search_type_str}, Q: {Q_str}")


def plt_convergence_rate(history, func: Function, opt_point: np.ndarray, title: str):
    # Plotting
    plt.figure(figsize=(10, 6))
    convergences = [func.f(x) - func.f(opt_point) for x in history]

    plt.plot(range(len(history)), convergences, marker='o', linestyle='-', color='b')
    plt.yscale('log')
    plt.xlabel('Iteration Number (k)')
    plt.ylabel('f(xk) - f*')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def test_rosenbrock(search_type: SearchType, starting_point: np.ndarray):
    search_type_str = "exact" if (search_type == SearchType.Exact) else "inexact"

    func = Rosenbrock2D()

    x, history = gradient_descent(func, starting_point, search_type)
    plot_line_search(func, history,
                     f"2D Rosenbrock gradient descent, {search_type_str}", display_log_norm_contours=True)
    plt_convergence_rate(history, func, np.array([1, 1]), "Convergence curve of gradient descent")

    x, history = newtons_method_line_search(func, starting_point, search_type, display_armijo=False)
    plot_line_search(func, history,
                     f"2D Rosenbrock Newtons method, {search_type_str}", display_log_norm_contours=True)
    plt_convergence_rate(history, func, np.array([1, 1]), "Convergence curve of Newtons method")


def main():
    starting_point1 = np.array([1.5, 2], dtype=float)
    Q1 = np.array([[3, 0], [0, 3]], dtype=float)
    Q2 = np.array([[10, 0], [0, 1]], dtype=float)
    test_quadratic_form(Q1, SearchType.Exact, starting_point1)
    test_quadratic_form(Q2, SearchType.Exact, starting_point1)
    test_quadratic_form(Q2, SearchType.Inexact, starting_point1)
    starting_point2 = np.array([0, 0], dtype=float)
    test_rosenbrock(SearchType.Inexact, starting_point2)


if __name__ == '__main__':
    main()
