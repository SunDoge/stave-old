from stave.nn.decorator import MODULE
from jax.interpreters.xla import DeviceArray
from stave import nn
from dataclasses import dataclass, field
import stave.nn.functional as F
import logging

_logger = logging.getLogger(__name__)


@nn.differentiable
@dataclass(repr=False)
class MLP(nn.Module):
    linear1: nn.Linear = field(metadata=nn.MODULE)
    linear2: nn.Linear = field(metadata=nn.MODULE)

    @classmethod
    def new(cls, in_features: int, hidden_dim: int, out_features: int):
        linear1 = nn.Linear.new(in_features, hidden_dim)
        linear2 = nn.Linear.new(hidden_dim, out_features)

        return cls(
            linear1=linear1,
            linear2=linear2,
        )

    def __call__(self, x: DeviceArray) -> DeviceArray:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


def main():

    logging.basicConfig(level=logging.INFO)

    in_features = 28 * 28
    hidden_dim = 1024
    out_features = 10

    model = MLP.new(in_features, hidden_dim, out_features)

    _logger.info('mlp: %s', model)


if __name__ == "__main__":
    main()
