from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as partial
from jax.tree_util import tree_map, tree_flatten

from models.modules.initializers import constant_value_initialiser
from models.modules.set_convolution import BatchedSetConvolution


def reshape_with_expanded_complex_axis(features, complex):
    """
    Reshapes a real-valued array s-as-to expand it with additional axis representing the complex plane.

    :param features:    Array to be reshaped.
    :param complex:     In {1, 2}. Size of added complex-plane axis.
                            1: real-valued data.
                            2: complex-valued data.
    :return:            Reshaped array with additional trailing axis representing the complex plane.
    """

    channels_ = features.shape[-1]

    if complex == 2:

        return jnp.stack((features[..., :channels_ // 2], features[..., channels_ // 2:]), axis=-1)

    else:

        assert complex == 1
        return jnp.expand_dims(features, axis=-1)


def get_mu_channel_count(total_channel_count, n_complex, isotropic_complex_noise):
    """
    recovers number of complex-valued or real-valued channel count for mu

    the count is the same for real-valued channel count for sigma

    total_channel_count = channel_count(mu) + channel_count(sigma)
                            = mu_channel_count * n_complex + channel_count(sigma)
                            = mu_channel_count * n_complex + mu_channel_count
                            = (n_complex + 1 ) * mu_channel_count
    """

    if isotropic_complex_noise:
        mu_channel_count = total_channel_count / (n_complex + 1)  # mu + real-valued sigma
    else:
        mu_channel_count = total_channel_count / (n_complex + n_complex)  # mu + real-valued sigma

    assert mu_channel_count == int(mu_channel_count)

    return int(mu_channel_count)


def f_2_mu(f, y_c, isotropic_complex_noise):
    n_complex = y_c.shape[-1]

    mu_channel_count = get_mu_channel_count(total_channel_count=f.shape[-1], n_complex=n_complex,
                                            isotropic_complex_noise=isotropic_complex_noise)

    mu = f[..., :mu_channel_count * n_complex]

    mu = reshape_with_expanded_complex_axis(mu, complex=n_complex)

    return mu


def f_2_sigma(f, y_c, isotropic_complex_noise):
    n_complex = y_c.shape[-1]

    mu_channel_count = get_mu_channel_count(total_channel_count=f.shape[-1], n_complex=n_complex,
                                            isotropic_complex_noise=isotropic_complex_noise)

    sigma = f[..., mu_channel_count * n_complex:]

    if isotropic_complex_noise:

        sigma = reshape_with_expanded_complex_axis(sigma, complex=1)

    else:

        sigma = reshape_with_expanded_complex_axis(sigma, complex=n_complex)

    return sigma


def broadcast_batch(value_to_broadcast, batch_size):
    return jax.lax.broadcast_in_dim(value_to_broadcast, shape=(batch_size, *value_to_broadcast.shape),
                                    broadcast_dimensions=tuple(range(1, len(value_to_broadcast.shape) + 1)))


class ConvCNP(nn.Module):
    """
    ConvCNP for off-the-grid data

    follows 'CONVOLUTIONAL CONDITIONAL NEURAL PROCESSES' by Gordon et al., 2020

    grid:       resolution x resolution x 3, a spherical grid in euclidian coordinates
    """
    grid: Sequence[float]
    input_set_convolution_module: BatchedSetConvolution
    cnn_module: nn.Module
    latent_set_convolution_module: BatchedSetConvolution
    pointwise_decoder_module: Callable
    mu_transform: Callable
    sigma_transform: Callable
    isotropic_complex_noise: bool
    calibration_set_convolution_module: BatchedSetConvolution

    @staticmethod
    def check_inputs(x_c, y_c, x_t, mask):

        # TODO: make sanity checks that allows for value inputs y of shape y.shape = (*x.shape, sequence, spin, channel, complex)
        assert len(x_t.shape) in [3, 4], 'expected array of shape (batch, context_data_points, 3) or ' \
                                         '(batch, resolution, resolution, 3)'
        assert len(x_c.shape) in [3, 4], 'expected array of shape (batch, context_data_points, 3) or ' \
                                         '(batch, resolution, resolution, 3)'

        @partial(partial, tree_map)
        def check_y(y):

            if mask is not None:
                y.shape[1] == mask.shape[1]

            assert y.shape[:-4] == x_c.shape[:-1], ' batch and spatial dimension(s) of context datapoint pair ' \
                                                   '(x_c, y_c) must agree'

            assert len(y.shape) in [6, 7], 'expected array of shape (batch_size, context_data_points, sequence, spins,' \
                                           'channels, real/imaginary part) or (batch_size, resolution, resolution, sequence,' \
                                           'spins, channels, real/imaginary part), where channel is typically the ear index'

            if len(y.shape) == 7:
                assert y.shape[1] == y.shape[2], 'expected array of shape (batch, resolution, ' \
                                                 'resolution, sequence, spins, channels, real/imaginary part)'

            assert y.shape[-1] in [1, 2], 'expected trailing dimension to be 1 (real input) or 2 (complex input)'

        check_y(y_c)

    @nn.compact
    def __call__(self, x_c, y_c, x_t, mask, *args, **kwargs):
        """
        Applies the network to inputs.

        Args:

            x_c:         cartesian coordinates of context data points, with shape (batch, *spatial, 3)

            y_c:         values of context data points, with shape (batch, *spatial, sequence, spins, channels,
                         real/imaginary part), where channel is typically the ear index

            x_t:         cartesian coordinates of target data points, with shape (batch, *spatial, 3)

            mask:        optional boolean array that specifies the active coordinates in x_c:
                            + mask is not None: allows batch elements with differing number of context points
                            + mask is None: all context points are assumed active for all batch elements

        where spatial denotes spatial dimensions, e.g. shape (# of datapoints,) for 1d or off-the-grid data, or
        (n_points_height, n_points width) for 2d data (in particular on-the grid)

        Returns:

            mu_t:         predicted mean of target data points y_t, with shape (batch_size, target_data_points,
                          sequence, spins, channels)

            sigma_t:      predicted standard deviation of target data points y_t, with shape (batch_size,
                          target_data_points, sequence, spins, channels)
        """

        ConvCNP.check_inputs(x_c, y_c, x_t, mask)

        def E(y, module_prefix, x, m):

            input_set_convolution = self.input_set_convolution_module(name=module_prefix + 'input_set_convolution')
            return input_set_convolution(mask=m,
                                         x_c=x,
                                         y_c=y,
                                         x_t=broadcast_batch(self.grid, y.shape[0]))

        if isinstance(y_c, dict):
            module_prefix = {key: key + '/' for key in y_c.keys()}
        else:
            module_prefix = ''

        h = tree_map(lambda z, prefix: E(z, prefix, x_c, mask), y_c, module_prefix)

        f = self.cnn_module(name='cnn')(h)

        ''' get output value at target locations '''

        # TODO: how do we name the following variable so that different from f_t, which is the output of the decoder?
        latent_set_convolution = self.latent_set_convolution_module(name='latent_set_convolution')
        fullmask = jnp.ones((f.shape[0], *self.grid.shape[:-1]), dtype=bool)
        f_t = latent_set_convolution(mask=fullmask,
                                     x_c=broadcast_batch(self.grid, f.shape[0]),
                                     y_c=reshape_with_expanded_complex_axis(f, complex=1),
                                     x_t=x_t)

        ''' final point-wise transformation '''

        f_t = self.pointwise_decoder_module(name='decoder')(f_t)

        ''' parse mu and sigma '''

        mu = tree_map(partial(f_2_mu, isotropic_complex_noise=self.isotropic_complex_noise), f_t, y_c)
        sigma = tree_map(partial(f_2_sigma, isotropic_complex_noise=self.isotropic_complex_noise), f_t, y_c)
        # sanity check
        leaves, _ = tree_flatten(tree_map(lambda z1, z2: np.all(z1.shape[:-1] == z2.shape[:-1]), sigma, mu))
        assert np.all(leaves)

        ''' calibration layer (the case being) '''
        if self.calibration_set_convolution_module is not None:
            assert not isinstance(sigma, dict), 'this case is not yet coded for dictionary feature case'
            sigma_gain_grid = self.param('sigma_gain_grid',
                                         partial(constant_value_initialiser, constant=1.0),
                                         (*self.grid.shape[:-1], *sigma.shape[-4:]))
            calibration_set_convolution = self.calibration_set_convolution_module(name='calibration_set_convolution')
            fullmask = jnp.ones(self.grid.shape[0:-1], dtype=bool)
            sigma_gain = []
            for i in range(sigma.shape[-1]):
                sigma_gain += [calibration_set_convolution(mask=broadcast_batch(fullmask, sigma.shape[0]),
                                                           x_c=broadcast_batch(self.grid, sigma.shape[0]),
                                                           y_c=broadcast_batch(sigma_gain_grid[..., i:i + 1],
                                                                               sigma.shape[0]),
                                                           x_t=x_t)]
            sigma_gain = jnp.stack(sigma_gain, axis=-1)
            sigma *= sigma_gain

        ''' apply specified transformations '''

        mu = tree_map(self.mu_transform, mu)
        sigma = tree_map(self.sigma_transform, sigma)

        # TODO: make this a block decorating ConvCNP
        ''' adding (dummy) leading sample dimension, so as to have the same output shape than LNP model counterpart '''

        mu = tree_map(lambda w: jnp.expand_dims(w, axis=0), mu)
        sigma = tree_map(lambda w: jnp.expand_dims(w, axis=0), sigma)

        return mu, sigma
