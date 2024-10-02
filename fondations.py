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


def tanh(x: ndarray) -> ndarray:
    return np.tanh(x)


from typing import Callable, List

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def chain_length_2(chain: Chain, a: ndarray) -> ndarray:
    assert len(chain) == 2, f"lenght of input 'chain' should be 2 but got {len(chain)}"
    f1, f2 = chain[0], chain[1]
    return f2(f1(a))

def chain_length_3(chain: Chain, a: ndarray) -> ndarray:
    assert len(chain) == 3 , f"lenght of input 'chain' should be 3 but got {len(chain)}"
    f1, f2, f3 = chain[0], chain[1], chain[2]
    return f3(f2(f1(a)))

def chain_deriv_2(chain:Chain ,input_range : ndarray)-> ndarray:
    assert len(chain) == 2, f"lenght of input 'chain' should be 2 but got {len(chain)}"
    f1, f2 = chain[0], chain[1]
    df1dx = deriv(f1, input_range)
    df2dx_of_f1 = deriv(f2, f1(input_range))
    return df1dx*df2dx_of_f1

def chain_deriv_3(chain:Chain ,input_range :ndarray)-> ndarray:
    assert len(chain) == 3 , f"lenght of input 'chain' should be 3 but got {len(chain)}"
    f1, f2, f3 = chain[0], chain[1], chain[2]
    f1of_x = f1(input_range)
    f2of_x = f2(f1(input_range))
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1of_x)
    df3du = deriv(f3, f2of_x)
    return df1dx*df2du*df3du


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    PLOT_RANGE = np.arange(-3, 3, 0.01)
    chain1 = [square, sigmoid]
    chain2 = [sigmoid , square , tanh]

    def plot_chain(chain: Chain, PLOT_RANGE : ndarray):
        values = np.array([ chain_length_3(chain, a) for a in PLOT_RANGE ])
        derivs = np.array([ chain_deriv_3(chain,a) for a in PLOT_RANGE ])
        plt.plot(PLOT_RANGE, values, label="plot chain")
        plt.plot(PLOT_RANGE, derivs, label="plot derivative")
        plt.legend(["$f'(x) = (\tanh(\sigma(x)^2))'$"])
        plt.xlabel("values")
        plt.ylabel("images")
        plt.grid()
        plt.show()

    plot_chain(chain2, PLOT_RANGE)