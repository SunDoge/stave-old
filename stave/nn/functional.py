from jax import numpy as jnp, jit, lax
from typing import Any, Optional, Sequence

from jax.interpreters.xla import DeviceArray
# from jax.numpy import DeviceArray

from jax.nn import (
    relu,
    relu6,
)

Tensor = DeviceArray

# from jax.interpreters.xla import DeviceArray


def tanh(input: DeviceArray) -> DeviceArray:
    return jnp.tanh(input)


def dense(data: Tensor, weight: Tensor) -> Tensor:
    return jnp.dot(data, weight.T)

def bias_add(data: Tensor, bias: Tensor) -> Tensor:
    return data + bias


def linear(input: DeviceArray, weight: DeviceArray, bias: Optional[DeviceArray]) -> DeviceArray:
    output = jnp.dot(input, weight.T)
    if bias is not None:
        output += bias
    return output


def conv_nd(
    input: DeviceArray, weight: DeviceArray, bias: Optional[DeviceArray],
    stride: Sequence[int], padding: Any,
):
    output = lax.conv_general_dilated(
        input,
        weight,
        stride,
        padding,
    )

    if bias is not None:
        output += bias

    return output
