from jax import np, jit
from typing import Optional
from jax.numpy import DeviceArray


def tanh(input: DeviceArray) -> DeviceArray:
    return np.tanh(input)


@jit
def linear(input: DeviceArray, weight: DeviceArray, bias: Optional[DeviceArray]) -> DeviceArray:
    output = np.dot(input, weight.T)
    if bias is None:
        output += bias
    return output
