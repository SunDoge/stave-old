from .. import functional as F
from .module import Module
from jax import np, DeviceArray


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = np.ones([out_features, in_features])
        if bias:
            self.bias = np.ones([out_features])
        else:
            self.bias = None

    def reset_parameters(self):
        pass

    def forward(self, input: DeviceArray):
        return F.linear(input, self.weight, self.bias)
