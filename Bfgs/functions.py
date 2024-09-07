from collections.abc import Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class Function:
    f: Callable  # computes function at point
    g: Callable  # computes gradient at point
    h: Callable  # computes hessian at point
    exact_line_search: Callable  # computes optimal step given point and direction


def rosenbrock_f(x: np.ndarray) -> float:
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(0, len(x) - 1))


def rosenbrock_g(x: np.ndarray) -> np.ndarray:
    g = np.zeros_like(x)
    g[0] = 400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
    for i in range(1, len(x) - 1):
        g[i] = -200 * (x[i - 1] ** 2 - x[i]) + 400 * x[i] * (x[i] ** 2 - x[i + 1]) + 2 * (x[i] - 1)
    g[len(x) - 1] = -200 * (x[len(x) - 2] ** 2 - x[len(x) - 1])
    return g


class Rosenbrock(Function):
    def __init__(self):
        self.f = rosenbrock_f
        self.g = rosenbrock_g
        self.h = NotImplemented
        self.exact_line_search = NotImplemented
