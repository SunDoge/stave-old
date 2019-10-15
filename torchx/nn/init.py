from jax import np
import numpy as onp


def _calculate_fan_in_and_fan_out(tensor: np.DeviceArray):
    dimensions = tensor.ndim
