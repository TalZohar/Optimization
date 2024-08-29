from collections import Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class Function:
    f: Callable  # computes function at point
    g: Callable  # computes gradient at point
    h: Callable  # computes hessian at point
    exact_line_search: Callable  # computes optimal step given point and direction


def quadratic_f(x: np.ndarray, Q: np.ndarray) -> float:
    return 0.5 * (x.T @ Q @ x)


def quadratic_g(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return 0.5 * (Q.T + Q) @ x


def quadratic_h(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return 0.5 * (Q.T + Q)


def quadratic_exact_line_search(x: np.ndarray, d: np.ndarray, Q: np.ndarray) -> float:
    val_d_x = d.T @ (Q + Q.T) @ x
    val_d_d = d.T @ (Q + Q.T) @ d
    return -1 * val_d_x / val_d_d


class QuadraticFunction(Function):
    def __init__(self, Q: np.ndarray):
        self.f = lambda x: quadratic_f(x, Q)
        self.g = lambda x: quadratic_g(x, Q)
        self.h = lambda x: quadratic_h(x, Q)
        self.exact_line_search = lambda x, d: quadratic_exact_line_search(x, d, Q)


def rosenbrock2d_f(x: np.ndarray) -> float:
    x1, x2 = x[0], x[1]
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def rosenbrock2d_g(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    return np.array([
        -2 * (1 - x1) - 400 * x1 * (x2 - x1 ** 2),
        200 * (x2 - x1 ** 2)
    ])


def rosenbrock2d_h(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    return np.array([
        [1200 * x1 ** 2 - 400 * x2 + 2, -400 * x1],
        [-400 * x1, 200]
    ])


class Rosenbrock2D(Function):
    def __init__(self):
        self.f = rosenbrock2d_f
        self.g = rosenbrock2d_g
        self.h = rosenbrock2d_h
        self.exact_line_search = NotImplemented
