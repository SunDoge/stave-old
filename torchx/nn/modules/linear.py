import jax
from jax.numpy import DeviceArray
import numpy as np

from .. import functional as F
# from ..parameter import Parameter
from .module import Module, differentiable
from jax import np as jnp

@differentiable
class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # self.weight = Parameter(np.empty([out_features, in_features]))
        # if bias:
        #     self.bias = Parameter(np.empty([out_features]))
        # else:
        #     self.bias = None
        self._parameters['weight'] = jnp.ones([out_features, in_features])
        if bias:
            self._parameters['bias'] = jnp.ones([out_features])
        else:
            self._parameters['bias'] = None

    def reset_parameters(self):
        pass

    def forward(self, input: DeviceArray):
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
