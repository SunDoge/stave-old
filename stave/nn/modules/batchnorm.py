from .module import Module, differentiable, Differentiable
from jax import numpy as np
from typing import Optional

@differentiable
class _NormBase(Module):

    weight: Differentiable
    bias: Differentiable

    running_mean: Optional[np.ndarray]
    running_var: Optional[np.ndarray]

    momentum: Optional[float]
    



@differentiable
class _BatchNorm(_NormBase):

    def forward(self, input):
