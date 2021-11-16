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
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from stave import nn


_logger = logging.getLogger(__name__)

Tensor = DeviceArray

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
    _logger.info('build mnist train dataset')
    train_ds = MNIST('data/mnist', download=True,
                     transform=FlattenAndCast(), train=True)
    _logger.info('build mnist test dataset')
    val_ds = MNIST('data/mnist', download=False,
                   transform=FlattenAndCast(), train=False)

    return train_ds, val_ds


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


def ForwardStep(model: nn.Model):

    @jax.jit
    def _forward_step(params: dict, buffers: dict, data: Tensor, target: Tensor):
        output, new_buffers = model.pure_forward(params, buffers, data)
        loss = cross_entropy(output, target)
        acc = accuracy(output, target)
        return loss, (acc, new_buffers)

    return _forward_step


dforward_step = jax.value_and_grad(forward_step, has_aux=True)
dforward_step = jax.jit(dforward_step)


@jax.jit
def sgd(param: DeviceArray, update: DeviceArray) -> DeviceArray:
    # print('param:', param.shape)
    return param - 0.01 * update


def to_device_array(*xs) -> List[DeviceArray]:
    return [jnp.array(x) for x in xs]


@jax.jit
def update_params(params: dict, grads: dict):
    return jax.tree_multimap(sgd, params, grads)

def train(model: nn.Model, train_loader: NumpyLoader, epoch: int) -> MLP:

    forward_step = ForwardStep(model)
    dforward_step = jax.value_and_grad(forward_step, has_aux=True)
    dforward_step = jax.jit(dforward_step)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = to_device_array(data, target)

        # Update model
        (loss, (acc, new_buffers)), grads = dforward_step(model.params, model.buffers, data, target)
        # import ipdb; ipdb.set_trace()

        # Update model again, with SGD optimization
        model.params = update_params(model.params, grads)
        model.buffers = new_buffers

        _logger.info(
            f'Train epoch {epoch} [{batch_idx}/{len(train_loader)}]\t'
            f'Loss: {loss}, Acc: {acc}'
        )

    return model


def test(model: nn.Model, test_loader: NumpyLoader, epoch: int):

    forward_step = ForwardStep(model)

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = to_device_array(data, target)

        # Don't update model
        loss, (acc, _new_buffers) = forward_step(model.params, model.buffers, data, target)

        _logger.info(
            f'Test epoch {epoch} [{batch_idx}/{len(test_loader)}]\t'
            f'Loss: {loss}, Acc: {acc}'
        )


def main():

    logging.basicConfig(level=logging.INFO, force=True)

    in_features = 28 * 28
    hidden_dim = 1024
    out_features = 10
    batch_size = 128
    num_epochs = 10

    # model = MLP.new(in_features, hidden_dim, out_features)
    model = MLP(in_features, hidden_dim, out_features).init()

    # print(model)
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
        train(model, train_loader, epoch)
        test(model, test_loader, epoch)

    with open('mlp.pkl', 'wb') as f:
        pickle.dump([model.params, model.buffers], f)
        _logger.info('save model')


if __name__ == "__main__":
    main()
