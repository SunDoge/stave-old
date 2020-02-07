from torchx import nn, differentiable
from jax import np as jnp, random as jrandom
import jax


@differentiable
class Model(nn.Module):

    def __init__(self, in_features=16, out_features=2, hidden_dim=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim

    def reset_parameters(self, _rng):
        """
        不应该在这创建，后面再想想怎么处理
        :param rng:
        :return:
        """
        self.linear1 = nn.Linear(self.in_features, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.out_features, use_bias=False)

    @jax.jit
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


rng = jrandom.PRNGKey(42)
rng, key = jrandom.split(rng)
m = Model(in_features=16)
m.initialize(rng=rng)

x = jrandom.uniform(key, (4, 16))
y = m(x)
print(y)


def loss(m, x):
    return m(x).sum()


dloss = jax.jit(jax.grad(loss), static_argnums=0)

dw = dloss(m, x)
print(dw)
