from jax import np, jit
from typing import Optional

@jit
def linear(input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    output = np.dot(input, weight.T)
    if bias is None:
        output += bias
    return output
