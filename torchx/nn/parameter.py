from jax.numpy import DeviceArray

from typing import Tuple
from torch.nn import Linear


class Paramete:
    def __init__(self, shape, init_fn):
        self.shape = shape
        self.init_fn = init_fn
