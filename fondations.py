import numpy as np
from numpy import ndarray, dtype
from functools import reduce

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


from typing import Callable, List, Tuple, Any

Array_Function = Callable[[ndarray], ndarray]
Chain = List[Array_Function]

def deriv(func: Callable[[ndarray], ndarray], input_: ndarray, delta: float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def chain_length_2(chain: Chain, a: ndarray) -> ndarray:
    assert len(chain) == 2, f"length of input 'chain' should be 2 but got {len(chain)}"
    f1, f2 = chain[0], chain[1]
    return f2(f1(a))

def chain_length_3(chain: Chain, a: ndarray) -> ndarray:
    assert len(chain) == 3 , f"length of input 'chain' should be 3 but got {len(chain)}"
    f1, f2, f3 = chain[0], chain[1], chain[2]
    return f3(f2(f1(a)))

def chain_deriv_2(chain:Chain ,input_range : ndarray)-> ndarray:
    assert len(chain) == 2, f"length of input 'chain' should be 2 but got {len(chain)}"
    f1, f2 = chain[0], chain[1]
    df1dx = deriv(f1, input_range)
    df2dx_of_f1 = deriv(f2, f1(input_range))
    return df1dx*df2dx_of_f1

def chain_deriv_3(chain:Chain ,input_range :ndarray)-> ndarray:
    assert len(chain) == 3 , f"length of input 'chain' should be 3 but got {len(chain)}"
    f1, f2, f3 = chain[0], chain[1], chain[2]
    f1of_x = f1(input_range)
    f2of_x = f2(f1(input_range))
    df1dx = deriv(f1, input_range)
    df2du = deriv(f2, f1of_x)
    df3du = deriv(f3, f2of_x)
    return df1dx*df2du*df3du

import numpy as np
from numpy import ndarray
from functools import reduce

def chain_length_n(chain: Chain, a: ndarray) -> ndarray:
    assert len(chain) != 0, f"length of input 'chain' should be different of 0 but got {len(chain)}"
    temp = chain[0](a)
    for i in range(1,len(chain)):
        temp = chain[i](temp)
    return temp

def chain_deriv_n(chain:Chain ,input_range :ndarray)-> ndarray:
    assert len(chain) != 0 , f"length of input 'chain' should be different of 0 but got {len(chain)}"
    derives : list[ndarray] = [deriv(chain[0], input_range)]
    temp_f : ndarray = input_range
    for i in range(1,len(chain)):
        temp_f = chain[i-1](temp_f)
        derives.append(deriv(chain[i], temp_f))
    return reduce(lambda x, y : x*y, derives)

def multiple_inputs_add_backward(x: ndarray, y: ndarray, sigma : Array_Function) -> tuple[ndarray, ndarray]:
    assert len(x) == len(y), f"length of input 'x' and 'y' should be equal but got {len(x)} and {len(y)}"
    a = x + y
    dsda = deriv(sigma, a)
    dadx , dady = 1 , 1
    return dsda*dadx , dsda*dady


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    assert X.shape[1] == W.shape[0] , f" number of columns of X should be equal to number of rows of W but got {X.shape[1]} and {W.shape[1]}"
    return np.dot(X,W)

def matmul_backward_first(X: ndarray, W: ndarray) -> ndarray:
    dNdX = np.transpose(W,(1,0))
    return dNdX

def matrix_forward_extra( X : ndarray, W : ndarray, sigma : Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0] , f" number of columns of X should be equal to number of rows of W but got {X.shape[1]} and {W.shape[1]}"
    N = np.dot(X,W)
    S = sigma(N)
    return S

def matrix_backward_1( X : ndarray, W : ndarray, sigma : Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0] , f" number of columns of X should be equal to number of rows of W but got {X.shape[1]} and {W.shape[1]}"
    N = np.dot(X,W)
    dSdu = deriv(sigma , N)
    dNdX = np.transpose(W, (1,0))

def matrix_function_forward_sum( X : ndarray, W : ndarray, sigma : Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0] , f" number of columns of X should be equal to number of rows of W but got {X.shape[1]} and {W.shape[1]}"
    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)
    return L

def matrix_function_backward_sum_1( X : ndarray, W : ndarray, sigma :Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0] , f" number of columns of X should be equal to number of rows of W but got {X.shape[1]} and {W.shape[1]}"
    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)

    dLdS = np.ones_like(S)
    dSdN = deriv(sigma , N)
    dLdN = dLdS * dSdN
    dNdX = np.transpose(W, (1,0))
    dLdX = np.dot(dSdN, dNdX)
    return dLdX

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """
        Example 1
    """
    """
        PLOT_RANGE = np.arange(-3, 3, 0.01)
    chain1 = [square, sigmoid]
    chain2 = [sigmoid , square , tanh]

    def plot_chain(chain: Chain, PLOT_RANGE : ndarray):
        values = np.array([ chain_length_n(chain, a) for a in PLOT_RANGE ])
        derivs = np.array([ chain_deriv_n(chain,a) for a in PLOT_RANGE ])
        plt.plot(PLOT_RANGE, values, label="plot chain")
        plt.plot(PLOT_RANGE, derivs, label="plot derivative")
        plt.legend(["$f(x) = (\\tanh(\sigma(x)^2))$","$f'(x) = (\\tanh(\sigma(x)^2))'$"])
        plt.xlabel("values")
        plt.ylabel("images")
        plt.title("Derivatives of order 3")
        plt.grid()
        plt.show()

    plot_chain(chain2, PLOT_RANGE)
    """

    """
        Example 2
    """

    np.random.seed(190204)
    X = np.random.randn(3,3)
    W = np.random.randn(3,3)
    print("X:")
    print(X)
    print("L:")

    print(round(matrix_function_forward_sum(X, W , sigmoid) , 4))

    print("\ndLdX")
    print(matrix_function_backward_sum_1(X, W , sigmoid))


