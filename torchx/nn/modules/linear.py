import jax
from jax.numpy import DeviceArray
import numpy as np

from .. import functional as F
from ..parameter import Parameter
from .module import Module


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(np.empty([out_features, in_features]))
        if bias:
            self.bias = Parameter(np.empty([out_features]))
        else:
            self.bias = None

    def reset_parameters(self):
        pass

    def forward(self, input: DeviceArray):
        return F.linear(input, self.weight, self.bias)
