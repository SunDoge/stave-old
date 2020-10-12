# stave

A proof-of-concept deep learning library based on [JAX](https://github.com/google/jax).

This project is inspired by [cgarciae/jax-differentiable](https://github.com/cgarciae/jax-differentiable).

**Note: I'm not good at naming, so the name may change in the future. This project is POC and WIP, everything will change.**

**TODO: API changed!!! Rewrite the README!!!**

Just checkout [examples/mnist.py](examples/mnist.py), the model is trainable now!

## Installation

### JAX installation

```bash
pip install --upgrade jax jaxlib  # CPU-only version
```

Check [google/jax](https://github.com/google/jax#pip-installation) for GPU version.

### stave installation

```bash
pip install git+https://github.com/SunDoge/stave.git
```

## Concepts

**Out-dated, checkout the example for the newest API!**

### Module

A module is a struct (dataclass in python). All attributes must be defined explicitly.

```python
from stave import nn, differentiable
from jax import np, jit

@differentiable
class Linear(nn.Module):
    weight: differentiable
    bias: differentiable

    @classmethod
    def new(cls, in_features, out_features):
        weight = np.empty([out_features, in_features])
        bias = np.empty([out_features])
        return cls(weight, bias)

    @jit
    def forward(self, x):
        return np.dot(x, self.weight) + self.bias
```

We use `differentiable` to mark the parameters. Unfortunately, we don't have macro of decorator for attribute in python, so I have to use the annotation, which is not a optimal design.

`@differetiable` will register your module automatically so that it can be used in `jax.jit` and `jax.grad`.

Here, we use `def new(cls, *args, **kwargs)` to create a module. The method name may change in the future. I choose `new` because of `Rust` and `Go`.

The computation happens in `forward` function. Maybe I'll use `__call__` function in the future for better type hint.

### Model

Model is a combination of modules.

```python
from stave import nn, differentiable
from jax import np, jit

@differentiable
class Model(nn.Module):
    linear1: nn.Linear
    linear2: nn.Linear

    @classmethod
    def new(cls, in_features, out_features, hidden_dim):
        linear1 = nn.Linear.new(in_features, hidden_dim)
        linear2 = nn.Linear.new(hidden_dim, out_features)
        return cls(linear1, linear2)

    @jit
    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        return out
```

Make sure you write correct annotations, because the `differentiable` depends on them.

### Gradient

To get the gradient of the model, we need to create a wrapper function, due to the design of `jax.grad`.

```python
import jax
from jax import np
from stave import nn

def loss_fn(output, target):
    return (output - target) ** 2

def loss_wrapper(model, input, target):
    output = model(input)
    L = loss_fn(output, input)
    return L

dloss = jax.jit(jax.grad(loss_wrapper))

model = nn.Linear.new(2, 4)
input = np.ones((32, 2))
label = np.ones((32,))

dmodel = dloss(model, input, label)
```

`dmodel` is an instance of model, but with all parameters replaced by their gradients.

### Optimizer

TODO
