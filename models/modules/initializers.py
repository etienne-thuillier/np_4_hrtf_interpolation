from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

_complex_zeros_initializer = partial(nn.initializers.zeros, dtype=jnp.complex64)

_complex_ones_initializer = partial(nn.initializers.ones, dtype=jnp.complex64)


def softplus_minus_1_scale_initialiser__von_mises(key, shape, dtype=jnp.float_):
    return -5.3 * jnp.ones(shape, jax.dtypes.canonicalize_dtype(dtype))


def constant_value_initialiser(key, shape, dtype=jnp.float_, constant=-1.0):
    return constant * jnp.ones(shape, jax.dtypes.canonicalize_dtype(dtype))


def softplus_inverse(y):
    """ knowing y, returns x in y = softplus(x) """

    return np.log(np.exp(y) - 1)
