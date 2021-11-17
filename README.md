# stave

A proof-of-concept deep learning library based on [JAX](https://github.com/google/jax).

## Installation

### JAX installation

```bash
pip install --upgrade "jax[cpu]"
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
from stave import nn
import stave.nn.functional as F

class MLP(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Seq(
            [nn.Dense(in_features, hidden_dim), nn.BiasAdd(hidden_dim)])
        self.linear2 = nn.Seq(
            [nn.Dense(hidden_dim, out_features), nn.BiasAdd(out_features)])

        # self.linear1 = nn.Dense(in_features, hidden_dim)
        # self.linear2 = nn.Dense(hidden_dim, out_features)

    def forward(self, x: DeviceArray) -> DeviceArray:
        x = self.linear1.forward(x)
        x = F.relu(x)
        x = self.linear2.forward(x)
        return x
```


### Model

`Model` is a container for `parameters`, `buffers` and `pure_forward` function.

```python

```

### Gradient

To get the gradient of the model, we need to create a wrapper function, due to the design of `jax.grad`.

```python
```


### Optimizer

TODO
