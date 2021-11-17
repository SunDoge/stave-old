
from timeit import timeit

import numpy as onp
import torch
import torch.nn as torchnn
torchnn.Sequential
from jax import random, grad, jit, numpy as np
import jax

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


@jit
def step1(m, x):
    return np.sum(m(x)), m

gstep1 = jax.value_and_grad(step1, has_aux=True)
gstep1 = jax.jit(gstep1)

(loss, grad), linear1 = gstep1(linear1, x1)

import ipdb; ipdb.set_trace()

