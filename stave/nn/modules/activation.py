from jax.interpreters.xla import DeviceArray
from ..decorator import differentiable
from dataclasses import dataclass
from .module import _Module
from .. import functional as F



@differentiable
@dataclass(repr=False)
class ReLU(_Module):

    def __call__(self, x: DeviceArray) -> DeviceArray:
        return F.relu(x)

    