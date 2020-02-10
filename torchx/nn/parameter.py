from jax.numpy import DeviceArray

from typing import Tuple, List, Union, Iterable, Optional
from torch.nn import Linear
from dataclasses import dataclass


@dataclass
class Parameter:
    shape: Iterable[int]
    data: Optional[DeviceArray] = None
