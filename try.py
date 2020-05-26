
from timeit import timeit

import numpy as onp
import torch
import torch.nn as torchnn
from jax import random, grad, jit, numpy as np

import torchx.nn as torchxnn

key = random.PRNGKey(0)
x1 = random.normal(key, (4, 16))
x2 = torch.from_numpy(onp.array(x1))

linear1 = torchxnn.Linear.new(16, 1)
linear1.initialize()

linear2 = torchnn.Linear(16, 1)

@jit
def step(m, x):
    return np.sum(m(x))

y = linear1(x1)
print(y)
# print(jit(grad(step))(linear1, x1))

# linear1.initialize()
# print(jit(grad(step))(linear1, x1))
gstep = grad(step)

print(gstep(linear1, x1))

print('*' * 50)
print('check if flatten unflatten cached')
print(gstep(linear1, x1))

y = linear2(x2)
print(y)

