from typing import List, Tuple

import numpy as np
from numpy import ndarray


def forward_pass(x, weights: List[ndarray], biases: List[ndarray]):
    layer_outputs = [x]
    pre_activations = []

    for W, b in zip(weights, biases):
        pre = np.dot(layer_outputs[-1], W) + b
        pre_activations.append(pre)
        post = np.tanh(pre)
        layer_outputs.append(post)

    return layer_outputs[-1], layer_outputs, pre_activations


def tanh(x: ndarray) -> ndarray:
    return np.tanh(x)


def tanh_derivative(x: ndarray) -> ndarray:
    return 1 - np.tanh(x) ** 2


def mse(predicted: float, target: float) -> float:
    return (predicted - target) ** 2


def mse_derivative(predicted: float, target: float) -> float:
    return 2 * (predicted - target)


def backpropagation(x: ndarray, y: float, weights: List[ndarray], biases: List[ndarray]) -> Tuple[
    List[ndarray], List[ndarray]]:
    """
    I didn't see a need to flatten the gradients
    """
    output, layer_outputs, pre_activations = forward_pass(x, weights, biases)
    dWs = [np.zeros_like(W) for W in weights]
    dbs = [np.zeros_like(b) for b in biases]

    # Compute gradient for the last layer
    dL_dF = mse_derivative(output, y)
    delta = dL_dF * tanh_derivative(pre_activations[-1])
    dWs[-1] = np.outer(layer_outputs[-2], delta)
    dbs[-1] = delta

    # Backpropagation for remaining layers
    for i in reversed(range(len(weights) - 1)):
        delta = np.dot(delta, weights[i + 1].T) * tanh_derivative(pre_activations[i])
        dWs[i] = np.outer(layer_outputs[i], delta)
        dbs[i] = delta

    return dWs, dbs


def compute_loss_for_dataset(X: ndarray, Y: List[float], weights: List[ndarray], biases: List[ndarray]) -> float:
    num_samples = X.shape[0]
    losses = []

    # Iterate over each training example
    for i in range(num_samples):
        x_i = X[i]
        y_i = Y[i]

        output, _, _ = forward_pass(x_i, weights, biases)
        loss = mse(output, y_i)
        losses.append(loss)

    return sum(losses) / num_samples


def compute_gradient_for_dataset(X: ndarray, Y: List[float], weights: List[ndarray], biases: List[ndarray]) -> Tuple[
    List[ndarray], List[ndarray]]:
    num_samples = X.shape[0]

    additive_dWs = [np.zeros_like(W) for W in weights]
    additive_dbs = [np.zeros_like(b) for b in biases]

    # Iterate over each training example
    for i in range(num_samples):
        x_i = X[i]
        y_i = Y[i]

        dWs, dbs = backpropagation(x_i, y_i, weights, biases)

        # Accumulate the gradients
        for l in range(len(weights)):
            additive_dWs[l] += dWs[l]
            additive_dbs[l] += dbs[l]

    # Compute the average gradients
    averaged_dWs = [dW / num_samples for dW in additive_dWs]
    averaged_dbs = [db / num_samples for db in additive_dbs]

    return averaged_dWs, averaged_dbs


def flatten_parameters(weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
    # Flatten each weight matrix and bias vector
    weight_vecs = [w.flatten() for w in weights]
    bias_vecs = [b.flatten() for b in biases]

    # Concatenate all flattened weight and bias vectors
    flattened_params = []
    for w_vec, b_vec in zip(weight_vecs, bias_vecs):
        flattened_params.extend(w_vec)
        flattened_params.extend(b_vec)

    return np.array(flattened_params)


def unpack_parameters(flattened_params: np.ndarray, layer_sizes: List[int]) -> Tuple[
    List[np.ndarray], List[np.ndarray]]:
    weights = []
    biases = []

    current_index = 0

    # Extract weights and biases
    for i in range(len(layer_sizes) - 1):
        # Define sizes for weight matrix and bias vector
        weight_size = layer_sizes[i] * layer_sizes[i + 1]
        bias_size = layer_sizes[i + 1]

        # Extract weight matrix
        W = flattened_params[current_index:current_index + weight_size]
        weight_matrix = W.reshape((layer_sizes[i], layer_sizes[i + 1]))
        weights.append(weight_matrix)
        current_index += weight_size

        # Extract bias vector
        b = flattened_params[current_index:current_index + bias_size]
        bias_vector = b.reshape((layer_sizes[i + 1],))
        biases.append(bias_vector)
        current_index += bias_size

    return weights, biases
