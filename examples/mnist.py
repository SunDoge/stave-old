import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import jax
import numpy as np
import stave.nn.functional as F
from jax import numpy as jnp
from jax.interpreters.xla import DeviceArray
from jax.nn import log_softmax
from stave import nn
from stave.nn import struct
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


_logger = logging.getLogger(__name__)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )


class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32))


def get_datasets():
    train_ds = MNIST('data/mnist', download=True,
                     transform=FlattenAndCast(), train=True)
    val_ds = MNIST('data/mnist', download=False,
                   transform=FlattenAndCast(), train=False)

    return train_ds, val_ds


@nn.differentiable
@dataclass(repr=False)
class MLP(nn._Module):
    linear1: nn._Linear = field(metadata=nn.PYTREE_NODE)
    linear2: nn._Linear = field(metadata=nn.PYTREE_NODE)

    @classmethod
    def new(cls, in_features: int, hidden_dim: int, out_features: int):
        linear1 = nn._Linear.new(in_features, hidden_dim)
        linear2 = nn._Linear.new(hidden_dim, out_features)

        return cls(
            linear1=linear1,
            linear2=linear2,
        )

    def __call__(self, x: DeviceArray) -> DeviceArray:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


def _one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def cross_entropy(outputs: DeviceArray, targets: DeviceArray) -> DeviceArray:
    probs = log_softmax(outputs)
    labels = _one_hot(targets, 10)
    loss = -jnp.mean(probs * labels)
    return loss


def accuracy(outputs: DeviceArray, targets: DeviceArray) -> DeviceArray:
    pred = jnp.argmax(outputs, axis=1)
    return jnp.mean(pred == targets)


@jax.jit
def forward_step(
    model: MLP,
    data: DeviceArray,
    target: DeviceArray,
) -> Tuple[DeviceArray, Tuple[MLP, DeviceArray]]:
    output = model(data)
    loss = cross_entropy(output, target)
    acc = accuracy(output, target)
    return loss, (model, acc)


dforward_step = jax.value_and_grad(forward_step, has_aux=True)
dforward_step = jax.jit(dforward_step)


@jax.jit
def sgd(param: DeviceArray, update: DeviceArray) -> DeviceArray:
    # print('param:', param.shape)
    return param - 0.01 * update


def to_device_array(*xs) -> List[DeviceArray]:
    return [jnp.array(x) for x in xs]


def train(model: MLP, train_loader: NumpyLoader, epoch: int) -> MLP:
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = to_device_array(data, target)

        # Update model
        (loss, (model, acc)), grads = dforward_step(model, data, target)
        # import ipdb; ipdb.set_trace()

        # Update model again, with SGD optimization
        model = jax.tree_multimap(sgd, model, grads)

        _logger.info(
            f'Train epoch {epoch} [{batch_idx}/{len(train_loader)}]\t'
            f'Loss: {loss}, Acc: {acc}'
        )

    return model


def test(model: MLP, test_loader: NumpyLoader, epoch: int):
    model.eval()

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = to_device_array(data, target)

        # Don't update model
        loss, (_model, acc) = forward_step(model, data, target)

        _logger.info(
            f'Test epoch {epoch} [{batch_idx}/{len(test_loader)}]\t'
            f'Loss: {loss}, Acc: {acc}'
        )


def main():

    logging.basicConfig(level=logging.INFO)

    in_features = 28 * 28
    hidden_dim = 1024
    out_features = 10
    batch_size = 128
    num_epochs = 10

    model = MLP.new(in_features, hidden_dim, out_features)
    model.initialize()

    _logger.info('mlp: %s', model)

    train_ds, test_ds = get_datasets()
    train_loader = NumpyLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = NumpyLoader(
        test_ds,
        batch_size=batch_size
    )

    for epoch in range(num_epochs):
        model = train(model, train_loader, epoch)
        test(model, test_loader, epoch)

    with open('mlp.pkl', 'wb') as f:
        pickle.dump(model, f)
        _logger.info('save model')


if __name__ == "__main__":
    main()
