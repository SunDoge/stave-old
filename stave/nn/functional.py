from jax import numpy as jnp, jit
from typing import Optional

from jax.interpreters.xla import DeviceArray
# from jax.numpy import DeviceArray

from jax.nn import (
    relu,
    relu6,
)


# from jax.interpreters.xla import DeviceArray


def tanh(input: DeviceArray) -> DeviceArray:
    return jnp.tanh(input)


def linear(input: DeviceArray, weight: DeviceArray, bias: Optional[DeviceArray]) -> DeviceArray:
    output = jnp.dot(input, weight.T)
    if bias is not None:
        output += bias
    return output
