import numpy as np
from numpy import ndarray


def square(x: ndarray) -> ndarray:
    return np.power(x, 2)


def leaky_relu(x: ndarray, alpha: float = 0.01) -> ndarray:
    return np.maximum(alpha * x, x)


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: ndarray) -> ndarray:
    exp = np.exp(x - np.max(x))
    return exp


def relu(x: ndarray) -> ndarray:
    return np.maximum(0, x)
