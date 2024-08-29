import numpy as np
import math
from typing import Callable
import matplotlib.pyplot as plt

VectorFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]
HessianFunction = Callable[[np.ndarray], np.matrix]

Function = Callable[[float], float]
FirstDerivativeFunction = Callable[[float], float]
SecondDerivativeFunction = Callable[[float], float]


def f1(x: np.ndarray, a: np.matrix, phi, ret_f=False, ret_g=False, ret_h=False):
    """
    Section 1.1.5
    Evaluates f1, its gradient, and Hessian at x, given ϕ (and its first and second orders).
    """
    f_value = g_value = h_value = None
    phi_function: VectorFunction = lambda x: phi(x, ret_f=True)[0]
    phi_gradient: GradientFunction = lambda x: phi(x, ret_g=True)[0]
    phi_hessian: HessianFunction = lambda x: phi(x, ret_h=True)[0]
    if ret_f:
        f_value = phi_function(a @ x)
    if ret_g:
        g_value = np.transpose(a) @ phi_gradient(a @ x)
    if ret_h:
        h_value = np.transpose(a) @ phi_hessian(a @ x) @ a

    return [x for x, keep in zip([f_value, g_value, h_value], [ret_f, ret_g, ret_h]) if keep]


def f2(x: np.ndarray, phi, h, ret_f=False, ret_g=False, ret_h=False):
    """
    Section 1.1.5
    Evaluates f2, its gradient, and Hessian at x, given ϕ (and its first and second orders) and h (and its first and second orders).
    """
    f_value = g_value = h_value = None
    phi_function: VectorFunction = lambda x: phi(x, ret_f=True)[0]
    phi_gradient: GradientFunction = lambda x: phi(x, ret_g=True)[0]
    phi_hessian: HessianFunction = lambda x: phi(x, ret_h=True)[0]
    h_function: Function = lambda x: h(x, ret_f=True)[0]
    h_first_der: FirstDerivativeFunction = lambda x: h(x, ret_g=True)[0]
    h_second_der: SecondDerivativeFunction = lambda x: h(x, ret_h=True)[0]

    if ret_f:
        f_value = h_function(phi_function(x))
    if ret_g:
        g_value = h_first_der(phi_function(x)) * phi_gradient(x)
    if ret_h:
        h_value = h_second_der(phi_function(x)) * phi_gradient(x) @ phi_gradient(x).transpose() + h_first_der(
            phi_function(x)) * phi_hessian(x)

    return [x for x, keep in zip([f_value, g_value, h_value], [ret_f, ret_g, ret_h]) if keep]


def phi(x: np.ndarray, ret_f=False, ret_g=False, ret_h=False):
    """
    Section 1.1.5
    Evaluates ϕ and its first and second derivatives
    """
    f_value = g_value = h_value = None
    x1, x2, x3 = x
    inner_value = x1 ** 2 + x2 * x3
    t_gradient = np.array([2 * x1, x3, x2])
    t_hessian = np.matrix([[2, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])
    if ret_f:
        f_value = math.sin(inner_value) ** 2
    if ret_g:
        g_value = math.sin(2 * inner_value) * t_gradient
    if ret_h:
        h_value = 2 * math.cos(2 * inner_value) * t_gradient * np.transpose(t_gradient) + math.sin(
            2 * inner_value) * t_hessian

    return [x for x, keep in zip([f_value, g_value, h_value], [ret_f, ret_g, ret_h]) if keep]


def h(x: float, ret_f=False, ret_g=False, ret_h=False):
    """
    Section 1.1.5
    Evaluates h and its first and second derivatives
    """
    f_value = g_value = h_value = None
    if ret_f:
        f_value = math.sqrt(1 + math.cos(x))
    if ret_g:
        g_value = -math.sin(x) / (2 * math.sqrt(1 + math.cos(x)))
    if ret_h:
        h_value = (-2 * math.cos(x) - 2 * math.cos(x) ** 2 - math.sin(x) ** 2) / (4 * (1 + math.cos(x)) ** (3 / 2))

    return [x for x, keep in zip([f_value, g_value, h_value], [ret_f, ret_g, ret_h]) if keep]


def numerical_evaluation(analytical_evaluation_func, x: np.ndarray, epsilon: float, *params):
    """
    Section 1.2.3
    Numerically evaluates the gradient and Hessian at x for a given function analytical_evaluation_func.
    """
    g_value = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        unit_vector = np.eye(1, len(x), i).reshape(-1, 1)
        fx_plus = analytical_evaluation_func(x + epsilon * unit_vector, *params, ret_f=True)[0]
        fx_minus = analytical_evaluation_func(x - epsilon * unit_vector, *params, ret_f=True)[0]
        g_value[i] = (fx_plus - fx_minus) / (2 * epsilon)

    h_value = np.zeros((len(x), 0), dtype=float)
    for i in range(len(x)):
        unit_vector = np.eye(1, len(x), i).reshape(-1, 1)
        gx_plus = analytical_evaluation_func(x + epsilon * unit_vector, *params, ret_g=True)[0]
        gx_minus = analytical_evaluation_func(x - epsilon * unit_vector, *params, ret_g=True)[0]
        i_column = (gx_plus - gx_minus) / (2 * epsilon)
        h_value = np.concatenate((h_value, i_column), axis=1)

    return g_value, h_value


def create_plot(x, y, name):
    # Create scatter plot
    plt.scatter(x, y, color='blue')

    plt.title(name)  # Optional: Add title
    plt.xlabel(f'analytical_{name}_norm')  # Optional: Add x-axis label
    plt.ylabel(f'analytical_numerical_{name}_distance')  # Optional: Add y-axis label

    plt.grid(True)  # Optional: Add grid
    plt.tight_layout()  # Optional: Adjust layout

    plt.savefig(f'{name}.png')
    plt.close()


MIN_RAND_VAL = 0
MAX_RAND_VAL = 4


def main():
    f1_results_hessian_norm = []
    f1_results_hessian_distance_norm = []
    f1_results_gradient_norm = []
    f1_results_gradient_distance_norm = []
    f2_results_hessian_norm = []
    f2_results_hessian_distance_norm = []
    f2_results_gradient_norm = []
    f2_results_gradient_distance_norm = []

    for i in range(1000):
        random_vector = np.random.rand(3, 1) * (MAX_RAND_VAL - MIN_RAND_VAL) + MIN_RAND_VAL
        random_matrix = np.random.rand(3, 3) * (MAX_RAND_VAL - MIN_RAND_VAL) + MIN_RAND_VAL
        # print(f"Random vector:\n{random_vector}\nRandom matrix:\n{random_matrix}")
        epsilon = (10 ** -6) * np.linalg.norm(random_vector, ord=np.inf)

        # comparison of f1 analytical and numerical distance
        analytical_f, analytical_g, analytical_h = f1(random_vector, random_matrix, phi, ret_f=True, ret_g=True,
                                                      ret_h=True)
        numerical_g, numerical_h = numerical_evaluation(f1, random_vector, epsilon, random_matrix, phi)
        g_distance = np.linalg.norm(analytical_g - numerical_g, ord=np.inf)
        h_distance = np.linalg.norm(analytical_h - numerical_h, ord=np.inf)
        analytical_g_norm = np.linalg.norm(analytical_g, ord=np.inf)
        numerical_g_norm = np.linalg.norm(numerical_g, ord=np.inf)
        analytical_h_norm = np.linalg.norm(analytical_h, ord=np.inf)
        numerical_h_norm = np.linalg.norm(numerical_h, ord=np.inf)
        # print(
        #     f"Analytical/Numberical f1 gradient Lmax: {analytical_g_norm}/{numerical_g_norm}")
        # print(
        #     f"Analytical/Numberical f1 hessian Lmax : {analytical_h_norm}/{numerical_h_norm}")
        # print(f"f1 gradient Lmax difference: {g_distance}. f1 Hessian Lmax difference {h_distance}\n")

        f1_results_gradient_norm.append(analytical_g_norm)
        f1_results_hessian_norm.append(analytical_h_norm)
        f1_results_gradient_distance_norm.append(g_distance)
        f1_results_hessian_distance_norm.append(h_distance)

        # comparison of f2 analytical and numerical distance
        analytical_f, analytical_g, analytical_h = f2(random_vector, phi, h, ret_f=True, ret_g=True, ret_h=True)
        numerical_g, numerical_h = numerical_evaluation(f2, random_vector, epsilon, phi, h)
        g_distance = np.linalg.norm(analytical_g - numerical_g, ord=np.inf)
        h_distance = np.linalg.norm(analytical_h - numerical_h, ord=np.inf)
        analytical_g_norm = np.linalg.norm(analytical_g, ord=np.inf)
        numerical_g_norm = np.linalg.norm(numerical_g, ord=np.inf)
        analytical_h_norm = np.linalg.norm(analytical_h, ord=np.inf)
        numerical_h_norm = np.linalg.norm(numerical_h, ord=np.inf)
        # print(
        #     f"Analytical/Numberical f2 gradient Lmax: {analytical_g_norm}/{numerical_g_norm}")
        # print(
        #     f"Analytical/Numberical f2 hessian Lmax : {analytical_h_norm}/{numerical_h_norm}")
        # print(f"f2 gradient Lmax difference: {g_distance}. f2 Hessian Lmax difference {h_distance}")

        f2_results_gradient_norm.append(analytical_g_norm)
        f2_results_hessian_norm.append(analytical_h_norm)
        f2_results_gradient_distance_norm.append(g_distance)
        f2_results_hessian_distance_norm.append(h_distance)

    create_plot(f1_results_gradient_norm, f1_results_gradient_distance_norm, "f1_gradient")
    create_plot(f1_results_hessian_norm, f1_results_hessian_distance_norm, "f1_hessian")
    create_plot(f2_results_gradient_norm, f2_results_gradient_distance_norm, "f2_gradient")
    create_plot(f2_results_hessian_norm, f2_results_hessian_distance_norm, "f2_hessian")


if __name__ == '__main__':
    main()
