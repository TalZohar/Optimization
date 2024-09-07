import matplotlib.pyplot as plt
import numpy as np

from functions import Function, Rosenbrock
from line_search import bfgs


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


def test_rosenbrock(starting_point: np.ndarray):
    n = len(starting_point)
    func = Rosenbrock()
    x, history = bfgs(func, starting_point)
    plt_convergence_rate(history, func, np.ones(n), "Convergence curve of BFGS")


def main():
    n = 10
    starting_point = np.zeros(n)
    test_rosenbrock(starting_point)


if __name__ == '__main__':
    main()
