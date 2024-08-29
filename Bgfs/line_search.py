from enum import Enum

from functions import Function
from mcholmz import modifiedChol

import numpy as np
import matplotlib.pyplot as plt

ARMIJO_ALPHA_0 = 1
ARMIJO_BETA = 0.5
ARMIJO_SIGMA = 0.25
CURVATURE_CONDITION_C = 0.9


def stopping_condition(func: Function, x) -> bool:
    gradient = func.g(x)
    norm = np.linalg.norm(gradient)
    return norm < 10 ** -5


def armijo_condition(func: Function, x, alpha: float, sigma: float, direction) -> bool:
    return func.f(x + alpha * direction) <= func.f(x) + sigma * alpha * np.dot(func.g(x), direction)


def curvature_condition(func: Function, x, alpha: float, c: float, direction) -> bool:
    return np.dot(func.g(x + alpha * direction).T, direction) > c * np.dot(func.g(x), direction)


def find_wolfe_step_size(func: Function, x, direction, alpha0, beta, sigma, c, display=False) -> float:
    alpha = alpha0
    history = [alpha]
    while not armijo_condition(func, x, alpha, sigma, direction) \
            or not curvature_condition(func, x, alpha, c, direction):
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


def bfgs(func: Function, starting_point):
    x = starting_point
    history = [x]
    I = np.eye(len(x))
    H = I  # The inverse hessian approximation

    while not stopping_condition(func, x):
        step_dir = -1 * H @ func.g(x)
        step_size = find_wolfe_step_size(func, x, step_dir, ARMIJO_ALPHA_0, ARMIJO_BETA, ARMIJO_SIGMA,
                                         CURVATURE_CONDITION_C)
        x_old = x
        x = x + step_size * step_dir
        history.append(x)

        # Compute change in x and gradient
        s = (x - x_old).reshape(-1, 1)
        y = (func.g(x) - func.g(x_old)).reshape(-1, 1)

        # Update inverse hessian approximation using the BFGS formula
        rho = 1.0 / (y.T @ s)
        H = (I - rho * (s @ y.T)) @ H @ (I - rho * (y @ s.T)) + rho * (s @ s.T)
        pass

    return x, history
