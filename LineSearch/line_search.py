from enum import Enum

from functions import Function
from mcholmz import modifiedChol

import numpy as np
import matplotlib.pyplot as plt

ARMIJO_ALPHA_0 = 1
ARMIJO_BETA = 0.5
ARMIJO_SIGMA = 0.25


class SearchType(Enum):
    Exact = 1
    Inexact = 2


def stopping_condition(func: Function, x) -> bool:
    gradient = func.g(x)
    norm = np.linalg.norm(gradient)
    return norm < 10 ** -5


def find_armijo_step_size(func: Function, x, direction, alpha0, beta, sigma, display=False) -> float:
    alpha = alpha0
    history = [alpha]
    while func.f(x + alpha * direction) > func.f(x) + sigma * alpha * np.dot(func.g(x), direction):
        alpha *= beta
        history.append(alpha)

    if display:
        X = np.linspace(0, 1, 100)
        vectorized_func = np.vectorize(lambda a: func.f(x + a * direction))
        Y = vectorized_func(X)
        plt.figure(figsize=(10, 6))
        plt.plot(X, Y)
        tangent = func.f(x) + X * np.dot(func.g(x), direction)
        armijo_line = func.f(x) + sigma * X * np.dot(func.g(x), direction)
        plt.plot(X, tangent)
        plt.plot(X, armijo_line)
        for alpha_val in history:
            plt.axvline(x=alpha_val, color='red', linestyle=':')
        title = f'point[{x}],dir[{direction}]'
        plt.title(title)
        plt.grid(True)
        plt.show()
        # plt.savefig(f'{title}.jpg')

    return alpha


def ldl_solve(gradient: np.ndarray, hessian: np.matrix):
    l, d, e = modifiedChol(hessian)
    # Ly = grad
    y = np.linalg.solve(l, gradient)
    # Dz = y
    z = np.divide(y, d.T).flatten()
    # L^t x = z
    return -1 * np.linalg.solve(l.T, z)


def gradient_descent(func: Function, starting_point, search_type: SearchType):
    x = starting_point
    history = [x]

    while not stopping_condition(func, x):
        step_dir = -1 * func.g(x)
        if search_type == SearchType.Exact:
            step_size = func.exact_line_search(x, step_dir)
        elif search_type == SearchType.Inexact:
            step_size = find_armijo_step_size(func, x, step_dir, ARMIJO_ALPHA_0, ARMIJO_BETA, ARMIJO_SIGMA)
        else:
            raise NotImplementedError

        x = x + step_size * step_dir
        history.append(x)

    return x, history


def newtons_method_line_search(func: Function, starting_point, search_type: SearchType, display_armijo=False):
    x = starting_point
    history = [x]

    while not stopping_condition(func, x):
        step_dir = ldl_solve(func.g(x), func.h(x))

        if search_type == SearchType.Exact:
            step_size = func.exact_line_search(x, step_dir)
        elif search_type == SearchType.Inexact:
            step_size = find_armijo_step_size(func, x, step_dir, ARMIJO_ALPHA_0, ARMIJO_BETA, ARMIJO_SIGMA, display=display_armijo)
        else:
            raise NotImplementedError

        x = x + step_size * step_dir
        history.append(x)

    return x, history
