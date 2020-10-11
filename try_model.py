from torchx import nn, differentiable
from jax import numpy as jnp, random as jrandom
import jax
from dataclasses import dataclass


@differentiable
class Model(nn.Module):
    # def __init__(self, in_features=16, out_features=2, hidden_dim=64):
    #     super().__init__()
    #     self.in_features = in_features
    #     self.out_features = out_features
    #     self.hidden_dim = hidden_dim
    linear1: nn.Linear
    linear2: nn.Linear

    @classmethod
    def new(cls, in_features=16, out_features=2, hidden_dim=64):
        linear1 = nn.Linear.new(in_features, hidden_dim)
        linear2 = nn.Linear.new(hidden_dim, out_features, use_bias=False)
        return cls(
            linear1=linear1,
            linear2=linear2
        )

    @jax.jit
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def train(self, mode=True):
        print('train mode')
        super().train(mode=mode)


rng = jrandom.PRNGKey(42)
rng, key = jrandom.split(rng)
m = Model.new(in_features=16)
m.initialize(rng=rng)
m.train()

# print(m)

x = jrandom.uniform(key, (4, 16))
y = m(x)
print(y)


def loss(m, x):
    return m(x).sum()


# dloss = jax.jit(jax.grad(loss))
dloss = jax.grad(loss)

dw = dloss(m, x)
print('dw =', dw)

# import ipdb; ipdb.set_trace()

dw = dloss(m, x)

# dw = dloss(m, x)

# dw = dloss(m, x)

# flat, tree = jax.tree_flatten(dw)
# dw1 = jax.tree_unflatten(tree, flat)
# print(dw1)
