import jax
from jax.numpy import DeviceArray
import numpy as np

from .. import functional as F
from ..parameter import Parameter
from .module import Module, differentiable
from jax import np as jnp, random as jrandom
from dataclasses import dataclass
from typing import Optional


# @differentiable
# @dataclass
# class Linear(Module):
#     # def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
#     #     super().__init__()
#     #     self.in_features = in_features
#     #     self.out_features = out_features
#     #     self.use_bias = use_bias
#
#     # self.weight = Parameter(np.empty([out_features, in_features]))
#     # if bias:
#     #     self.bias = Parameter(np.empty([out_features]))
#     # else:
#     #     self.bias = None
#
#     def reset_parameters(self, rng):
#         k1, k2 = jrandom.split(rng)
#         self._parameters['weight'] = jrandom.normal(k1, [self.out_features, self.in_features])
#         if self.use_bias:
#             self._parameters['bias'] = jrandom.normal(k2, [self.out_features])
#         else:
#             self._parameters['bias'] = None
#
#     def forward(self, input: DeviceArray):
#         return F.linear(input, self.weight, self.bias)
#
#     def __repr__(self):
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )


@differentiable
class Linear(Module):
    in_features: int
    out_features: int
    use_bias: bool

    weight: Parameter
    bias: Parameter

    def __call__(self, input: DeviceArray):
        return F.linear(input, self.weight, self.bias)

    def reset_parameters(self, rng):
        k1, k2 = jrandom.split(rng)
        self.weight = jrandom.normal(k1, self.weight.shape)
        if self.use_bias:
            self.bias = jrandom.normal(k2, self.bias.shape)

    @classmethod
    def build(cls, in_features: int, out_features: int, use_bias=True):
        weight = Parameter([out_features, in_features])
        if use_bias:
            bias = Parameter([out_features])
        else:
            bias = None
        return cls(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            weight=weight,
            bias=bias
        )


