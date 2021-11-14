from .module import Module, Parameter, Tensor


class BiasAdd(Module):

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = Parameter((num_features,))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.bias
