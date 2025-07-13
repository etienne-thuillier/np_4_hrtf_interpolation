import os
from functools import partial
from typing import Any, Callable
from typing import Any as KeyArray

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
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


def permutation_from_A_to_B(A: np.ndarray, B: np.ndarray, tol: float = 0.0):
    """
    Return `p` such that  B[p[i]]  ==  A[i]   (within `tol`, if given).

    Parameters
    ----------
    A, B : (N, 3) float arrays
        Same set of points, scrambled.
    tol : float, optional
        Acceptable Euclidean distance between matching points.
        Leave at 0 for exact, bit-wise comparison (fastest).

    Returns
    -------
    p : (N,) int array
        Index mapping from A to B.
    """
    if tol == 0.0:
        # ------- exact match, fast ---------- #
        # 1.  Sort both arrays lexicographically; keep original indices.
        sa = np.lexsort(A.T[::-1])      # sort by x, then y, then z
        sb = np.lexsort(B.T[::-1])

        # (optional sanity check)
        if not np.array_equal(A[sa], B[sb]):
            raise ValueError("A and B don't contain identical points.")

        # 2.  Build permutation:  p[i] = index in B that matches A[i]
        p = np.empty_like(sa)
        p[sa] = sb
        return p

    # ------- inexact match (e.g. small FP noise) ---------- #
    # Use a nearest-neighbour search.
    tree = cKDTree(B)
    d, p = tree.query(A, distance_upper_bound=tol)

    if (d > tol).any():
        raise ValueError(
            f"Some points were farther than tol={tol:g} (max distance {d.max():g})."
        )
    return p


def lebedev_degree_2_order(quadrature_grid_degree):
    table = {degree: order
             for degree, order
             in zip([6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350,
                     434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470,
                     3890, 4334, 4802, 5294, 5810],
                    [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47,
                     53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131])}

    if quadrature_grid_degree not in table.keys():
        raise ValueError(f"Degree {quadrature_grid_degree} not in Lebedev table.")

    return table[quadrature_grid_degree]


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
