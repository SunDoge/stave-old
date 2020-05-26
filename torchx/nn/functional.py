from jax import numpy as np, jit
from typing import Optional
from jax.numpy import DeviceArray


# from jax.interpreters.xla import DeviceArray


def tanh(input: np.ndarray) -> DeviceArray:
    return np.tanh(input)


def linear(input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    output = np.dot(input, weight.T)
    if bias is not None:
        output += bias
    return output
