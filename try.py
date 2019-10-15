from torchx.nn.functional import linear
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (5000, 5000))
