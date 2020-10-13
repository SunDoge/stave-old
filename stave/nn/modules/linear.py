from typing import Optional

from jax import numpy as jnp
from jax import random as jrandom
from jax.interpreters.xla import DeviceArray

from .. import functional as F
# from ..decorator import BUFFER, PARAMETER, differentiable

from .module import Module
from ..struct import differentiable, PYTREE_NODE
from dataclasses import dataclass, field


@differentiable
@dataclass(repr=False)
class Linear(Module):
    in_features: int
    out_features: int
    use_bias: bool

    weight: DeviceArray = field(metadata=PYTREE_NODE)
    bias: Optional[DeviceArray] = field(metadata=PYTREE_NODE)

    def __call__(self, input: DeviceArray):
        return F.linear(input, self.weight, self.bias)

    def _reset_parameters(self, rng: DeviceArray):
        k1, k2 = jrandom.split(rng)
        self.weight = jrandom.normal(k1, self.weight.shape)
        if self.bias is not None:
            self.bias = jrandom.normal(k2, self.bias.shape)

        # print('pppp')

    @classmethod
    def new(cls, in_features: int, out_features: int, use_bias=True):
        weight = jnp.empty([out_features, in_features])
        if use_bias:
            bias = jnp.empty([out_features])
        else:
            bias = None
        return cls(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            weight=weight,
            bias=bias
        )

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == "__main__":
    linear = Linear.new(5, 10)
    print(linear)
