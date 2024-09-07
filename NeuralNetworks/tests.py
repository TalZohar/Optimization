import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from NeuralNetworks.ANN import flatten_parameters, compute_gradient_for_dataset, unpack_parameters, \
    compute_loss_for_dataset, forward_pass


def generate_training_dataset(n: int, func: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Generate n random points uniformly distributed between (-2, -2) and (2, 2)
    X = np.random.uniform(low=-2.0, high=2.0, size=(n, 2))
    Y = np.array([func(x) for x in X])
    return X, Y


def generate_test_dataset(n: int) -> np.ndarray:
    # Generate n random points uniformly distributed between (-2, -2) and (2, 2)
    X = np.random.uniform(low=-2.0, high=2.0, size=(n, 2))
    return X


def initialize_model_parameters(layer_sizes: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    weights = []
    biases = []

    for i in range(1, len(layer_sizes)):
        m = layer_sizes[i - 1]
        n = layer_sizes[i]

        # Initialize weights with standard normal distribution and divide by sqrt(n)
        W = np.random.randn(m, n) / np.sqrt(n)
        weights.append(W)

        # Initialize biases with zeros
        b = np.zeros((n,))
        biases.append(b)

    return weights, biases


def plot_function_and_data(title: str,
                           f: Callable[[np.ndarray], np.ndarray],
                           data_points: np.ndarray,
                           ground_truth: np.ndarray,
                           show_scatter: bool) -> None:
    # Create a meshgrid for plotting the surface
    x = np.arange(-2, 2, 0.2)
    y = np.arange(-2, 2, 0.2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Evaluate the function over the meshgrid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))[0]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    if show_scatter:
        ax.scatter(data_points[:, 0], data_points[:, 1], ground_truth, color='r', s=50)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('f(X1, X2)')
    ax.set_title(title)
    # plt.show()
    plt.savefig(f"{title}.jpg")


# Importing stuff from BGFS project
import sys, os

sys.path.append(os.path.abspath(os.path.join('..', 'Bfgs')))

from Bfgs.functions import Function
from Bfgs.line_search import bfgs
import Bfgs.line_search


def function_to_approximate(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    return np.array([x1 * np.exp(-(x1 ** 2 + x2 ** 2))])


class TargetFunction(Function):
    def __init__(self, X, Y, layer_sizes: List[int]):
        self.f = lambda w: compute_loss_for_dataset(X, Y, *unpack_parameters(w, layer_sizes))
        self.g = lambda w: flatten_parameters(*compute_gradient_for_dataset(X, Y, *unpack_parameters(w, layer_sizes)))
        self.h = NotImplemented
        self.exact_line_search = NotImplemented


def main():
    # Generate training dataset
    n_train_samples = 500
    n_test_samples = 200
    layer_sizes = [2, 4, 3, 1]
    X_train, Y_train = generate_training_dataset(n_train_samples, function_to_approximate)
    X_test, Y_test = generate_training_dataset(n_test_samples, function_to_approximate)
    weights, biases = initialize_model_parameters(layer_sizes)
    w = flatten_parameters(weights, biases)

    func = TargetFunction(X_train, Y_train, layer_sizes)
    for epsilon in [10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4]:
        Bfgs.line_search.STOPPING_EPSILON = epsilon
        opt_w, history = bfgs(func, w)
        weights, biases = unpack_parameters(opt_w, layer_sizes)
        approx_func = lambda x: forward_pass(x, weights, biases)
        training_loss = compute_loss_for_dataset(X_train, Y_train, weights, biases)
        testing_loss = compute_loss_for_dataset(X_test, Y_test, weights, biases)
        print(
            f"Epsilon: {epsilon}, training loss: {training_loss}, testing_loss: {testing_loss}, steps: {len(history)}")
        title = f"epsilon[{epsilon}]"
        plot_function_and_data(title, approx_func, X_test, Y_test, True)


if __name__ == '__main__':
    main()
