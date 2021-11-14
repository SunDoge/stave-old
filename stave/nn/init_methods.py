

from typing import Any, Sequence, Tuple
from jax.interpreters.xla import DeviceArray
import math
import jax.numpy as jnp
import jax.random as jrandom

Tensor = DeviceArray


def _calculate_fan_in_and_fan_out(shape: Sequence[int]) -> Tuple[int, int]:
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for s in shape:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class InitMethod:

    def __call__(self, key: Tensor, shape: Sequence[int], dtype: Any) -> Tensor:
        pass


class XavierUniform(InitMethod):

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def __call__(self, key: Tensor, shape: Sequence[int], dtype: Any) -> Tensor:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out))
        # Calculate uniform bounds from standard deviation
        a = math.sqrt(3.0) * std
        return jrandom.uniform(key, shape=shape, dtype=dtype, minval=-a, maxval=a)
