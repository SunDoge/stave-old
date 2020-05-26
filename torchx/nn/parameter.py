# from jax.numpy import DeviceArray
from jax import numpy as np
from typing import Tuple, List, Union, Iterable, Optional, NewType
from torch.nn import Linear
from dataclasses import dataclass

Parameter = NewType('Parameter', np.ndarray)

#
# @dataclass
# class Parameter:
#     shape: Iterable[int]
#     data: Optional[DeviceArray] = None
