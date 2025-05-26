import os
from functools import partial
from typing import Any, Callable
from typing import Any as KeyArray

import jax.numpy as jnp
import numpy as np
from flax import core, struct
from flax.training import train_state
from jax.tree_util import tree_map
from omegaconf import OmegaConf


class TrainState(train_state.TrainState):
    epoch: int
    batch_stats: core.FrozenDict[str, Any]
    rng: KeyArray
    eval_fn: Callable = struct.field(pytree_node=False)


def remove_dir_recursively(path):
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
    os.rmdir(path)


def flatten_dictionary(d, key_prefix=None):
    l = dict()
    for key, item in d.items():

        if key_prefix is not None:
            key = '/'.join([key_prefix, key])

        if isinstance(item, dict):

            l.update(flatten_dictionary(item, key_prefix=key))

        else:

            l.update({key: item})

    return l


@partial(partial, tree_map)
def collapse_complex_axis(z):
    if z.shape[-1] == 2:

        return z[..., 0] + 1j * z[..., 1]

    elif z.shape[-1] == 1:

        return z[..., 0]

    else:

        assert False, 'real / image dimension must be of dimension in [1, 2]'


@partial(partial, tree_map)
def expand_complex_axis(x):
    if jnp.iscomplexobj(x):

        return jnp.stack((x.real, x.imag), axis=-1)

    else:

        return jnp.expand_dims(x, axis=-1)


def register_resolvers():
    OmegaConf.register_new_resolver("div", lambda x, y: x / y)
    OmegaConf.register_new_resolver("idiv", lambda x, y: x // y)
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver("times", lambda x, y: x * y)


def hermitian_pad(y, padding):
    assert y.shape[-3:] in [(1, 2, 1), (1, 2, 2)], 'expecting trailing (spin, channel, complex) dimension to be of ' \
                                                   'shape (1, 2, 1) or (1, 2, 2)'
    assert y.shape[-4] % 2 == 1, 'expecting odd frequency dimension'

    left_pad = y[..., 1:padding + 1, :, :, :]
    left_pad = jnp.flip(left_pad, axis=-4)
    if left_pad.shape[-1] == 2:
        left_pad = left_pad.at[..., 1].set(-left_pad[..., 1])

    right_pad = y[..., -(padding + 1):-1, :, :, :]
    right_pad = jnp.flip(right_pad, axis=-4)
    if right_pad.shape[-1] == 2:
        right_pad = right_pad.at[..., 1].set(-right_pad[..., 1])

    return jnp.concatenate((left_pad, y, right_pad), axis=-4)


def truncate_hermitian_pad(y, padding):
    assert y.shape[-3:] in [(1, 2, 1), (1, 2, 2)], 'expecting trailing (spin, channel, complex) dimension to be of ' \
                                                   'shape (1, 2, 1) or (1, 2, 2)'
    assert y.shape[-4] % 2 == 1, 'expecting odd frequency dimension'

    return y[..., padding:-padding, :, :, :]


def hermitian_pad_demo():
    import matplotlib.pyplot as plt
    n = 17
    padding = 5

    real = np.cos(np.arange(n) * 2 * np.pi / (n - 1))
    imag = np.sin(np.arange(n) * 2 * np.pi / (n - 1))

    y = np.stack((real, imag), axis=-1)
    y = np.stack((y, y), axis=-2)
    y = np.expand_dims(y, axis=-3)

    y_hat = hermitian_pad(y, padding=padding)
    y_hat_hat = truncate_hermitian_pad(y_hat, padding=padding)

    plt.plot(y[:, 0, 0, 0], 'b+-')
    plt.plot(np.arange(-padding, n + padding), y_hat[:, 0, 0, 0], 'bx:')
    plt.plot(y[:, 0, 0, 1], 'r+-')
    plt.plot(np.arange(-padding, n + padding), y_hat[:, 0, 0, 1], 'rx:')
    plt.grid()
    plt.show()

    plt.plot(y[:, 0, 0, 0], 'b+-')
    plt.plot(y_hat_hat[:, 0, 0, 0], 'bx:')
    plt.plot(y[:, 0, 0, 1], 'r+-')
    plt.plot(y_hat_hat[:, 0, 0, 1], 'rx:')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    hermitian_pad_demo()
