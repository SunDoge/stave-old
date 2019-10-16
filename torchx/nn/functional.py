from jax import np, jit
from typing import Optional
from jax.numpy import DeviceArray
# from jax.interpreters.xla import DeviceArray



def tanh(input: DeviceArray) -> DeviceArray:
    return np.tanh(input)


def linear(input: DeviceArray, weight: DeviceArray, bias: Optional[DeviceArray]) -> DeviceArray:
    output = np.dot(input, weight.T)
    if bias is None:
        output += bias
    return output
