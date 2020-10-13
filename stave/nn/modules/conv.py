from typing import Any, Optional, Sequence, Tuple

from jax.interpreters.xla import DeviceArray
from .module import Module
from ..struct import differentiable, PYTREE_NODE
from dataclasses import dataclass, field
from jax import random as jrandom
from jax import lax
from .. import functional as F


@differentiable
@dataclass(repr=False)
class _ConvNd(Module):
    _in_channels: int
    out_channels: int
    kernel_size: Sequence[int]
    stride: Sequence[int]
    padding: Any
    dilation: Sequence[int]

    weight: DeviceArray = field(metadata=PYTREE_NODE)
    bias: Optional[DeviceArray] = field(metadata=PYTREE_NODE)

    dimension_numbers: Tuple[str, str, str] = (
        'NHWC', 'HWIO', 'NHWC'
    )

    def _reset_parameters(self, rng: DeviceArray):
        k1, k2 = jrandom.split(rng)
        self.weight = jrandom.normal(k1, self.weight.shape)
        if self.bias is not None:
            self.bias = jrandom.normal(k2, self.bias.shape)

    def __call__(self, input: DeviceArray) -> DeviceArray:
        return F.conv_nd(
            input, self.weight, self.bias, self.stride, self.padding
        )


class Conv2d(_ConvNd):
    pass
