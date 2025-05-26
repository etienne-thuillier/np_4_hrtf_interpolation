from functools import partial
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map

from models.modules.initializers import softplus_minus_1_scale_initialiser__von_mises
from spin_spherical_cnns.layers import Initializer


class SquaredExponentialKernel:

    @staticmethod
    def metric(x1, x2):
        return ((x1 - x2) ** 2).sum(-1)

    @staticmethod
    def log_k(metric, scale):
        return - metric / (scale ** 2 + jnp.finfo(scale.dtype).eps)

    @staticmethod
    def k(metric, scale):
        log_k = SquaredExponentialKernel.log_k(metric, scale)
        return jnp.exp(log_k)


def squared_exponential_kernel(x1, x2, scale):
    metric = SquaredExponentialKernel.metric(x1, x2)
    return SquaredExponentialKernel.k(metric, scale)


def von_mises_kernel(x1, x2, scale):
    metric = SphericalGaussianKernel.metric(x1, x2)
    return SphericalGaussianKernel.k(metric, scale)


class SphericalGaussianKernel:
    """
    Von Spherical Gaussian kernel computed as:

        jnp.exp(jnp.clip(jnp.dot(x1, x2) - 1.0, a_max=0.0) / (scale + jnp.finfo(scale.dtype).eps))

    cf. see "Positive definite kernels: past, present and future", 2011, Fasshauer
    """

    @staticmethod
    def metric(x1, x2):
        """
            silently assumes ||xi|| == 1, forall i in {1, 2}
                + subtraction by one here prevents overflow when scale is close to 0
                + clipping avoids eventual numerical round-off overflow over 0 in case when dot product outputs value is
                  close to 1.0

            the returned value is in [0, 2], and is 0 for x1 == x2 synonymous with smallest possible distance
        """

        return - jnp.clip(jnp.dot(x1, x2) - 1.0, a_max=0.0)

    @staticmethod
    def log_k(metric, scale):
        return - metric / (scale + jnp.finfo(scale.dtype).eps)

    @staticmethod
    def k(metric, scale):
        return jnp.exp(SphericalGaussianKernel.log_k(metric, scale))


class ExponentialKernelSetConvolution(nn.Module):
    initializer: Optional[Initializer] = softplus_minus_1_scale_initialiser__von_mises
    kernel: Any = SphericalGaussianKernel
    return_density: Optional[bool] = False
    share_scale_across_data_and_density: Optional[bool] = None
    small_memory_footprint: Optional[bool] = False

    @staticmethod
    def regularise(metric, axis, mask):
        """ ... """

        # # find minimum distance along axis amongst non-masked values
        # # => deactivate masked values by over-writing them with maximum distances along axis
        # maxi =  lax.stop_gradient(jnp.max(metric, axis=axis, keepdims=True))
        # metric = mask * metric + ~mask * maxi
        # # => now find minimum distance along axis
        # mini = lax.stop_gradient(jnp.min(metric, axis=axis, keepdims=True))
        # # overwrite masked values with minimum distance for numerical stability (we want to avoid nan or inf even if masked)
        # metric = mask * metric + ~mask * mini

        minus_metric = - metric

        ''' overwrite masked values with axis minimum  '''
        mini = jnp.min(minus_metric, axis=axis, keepdims=True)
        minus_metric = mask * minus_metric + ~mask * lax.stop_gradient(mini)

        ''' get max '''
        maxi = jnp.max(minus_metric, axis=axis, keepdims=True)

        ''' overwrite masked values with overall maximum for numerical stability '''
        minus_metric = mask * minus_metric + ~mask * lax.stop_gradient(maxi)

        minus_metric -= lax.stop_gradient(maxi)

        return - minus_metric

    def metric(self, mask, x_c, x_t, axis):

        metric = jax.vmap(self.kernel.metric, in_axes=(None, 0), out_axes=-1)

        if len(x_c.shape) == 3:

            metric = jax.vmap(metric, in_axes=(None, 0), out_axes=-2)

        else:

            assert len(x_c.shape) == 2

        d = metric(x_t, x_c)
        d_reg = ExponentialKernelSetConvolution.regularise(metric=d, axis=axis, mask=mask)

        return d, d_reg

    @staticmethod
    def get_axes(mask):

        if len(mask.shape) == 2:  # apply dot product to pair of last dimensions of K

            K_axes = (-2, -1)
            y_c_axes = (0, 1)

        else:

            assert len(mask.shape) == 1
            K_axes = (-1,)
            y_c_axes = (0,)

        return K_axes, y_c_axes

    @staticmethod
    def normalise(k, axis):

        ''' because of above treatment of log_K, K is 0 if mask is False everywhere => avoid division by 0 '''
        denominator = jnp.sum(k, axis=axis, keepdims=True)
        not_zero = denominator != 0
        denominator = not_zero * denominator + ~not_zero * jnp.ones_like(denominator)

        return k / denominator

    @staticmethod
    def small_space_complexity_kernel_filter(kernel, scale, mask, metric, y_c):

        in_spatial_shape = mask.shape
        out_spatial_shape = metric.shape[:-len(in_spatial_shape)]

        if len(out_spatial_shape) == 1:
            def f(carry_1, metric_1):

                k = kernel.k(metric_1, scale)
                axes = tuple(range(len(in_spatial_shape)))
                return None, jnp.tensordot(mask * k, y_c, axes=[axes, axes])

        else:
            assert len(out_spatial_shape) == 2

            def f(carry_1, metric_1):

                def g(carry_2, metric_2):
                    k = kernel.k(metric_2, scale)
                    axes = tuple(range(len(in_spatial_shape)))
                    return None, jnp.tensordot(mask * k, y_c, axes=[axes, axes])

                return jax.lax.scan(g, init=None, xs=metric_1)

        _, y_t = jax.lax.scan(f, init=None, xs=metric)

        return y_t

        # if len(in_spatial_shape) == 2:
        #
        #     def body_fun(i, val):
        #
        #         j = i // in_spatial_shape[0]
        #         l = i % in_spatial_shape[0]
        #
        #         k = kernel.k(metric[..., j, l], scale)
        #
        #         k = jax.lax.broadcast_in_dim(operand=k,
        #                                      shape=(*k.shape, *[1] * len(y_c.shape[2:])),
        #                                      broadcast_dimensions=tuple(range(len(k.shape))))
        #
        #         return val + k * mask[j, l] * y_c[j, l, ...]
        #
        # else:
        #
        #     assert len(in_spatial_shape) == 1, 'covering only cases where the input spatial shape is 1d or 2d for now'
        #
        #     mask = jax.lax.broadcast_in_dim(operand=mask,
        #                                     shape=(*mask.shape, *[1] * len(y_c.shape[1:])),
        #                                     broadcast_dimensions=tuple(range(len(mask.shape))))
        #
        #
        #     def body_fun(i, val):
        #
        #         k = kernel.k(metric[..., i], scale)
        #
        #         k = jax.lax.broadcast_in_dim(operand=k,
        #                                      shape=(*k.shape, *[1] * len(y_c.shape[1:])),
        #                                      broadcast_dimensions=tuple(range(len(k.shape))))
        #
        #         return val + k * mask[i] * y_c[i, ...]
        #
        #
        # return jax.lax.scan(lower=0,
        #                     upper=jnp.prod(jnp.array(in_spatial_shape, dtype=int)),
        #                     body_fun=body_fun,
        #                     init_val=jnp.zeros(shape=(*out_spatial_shape,
        #                                               *y_c.shape[len(in_spatial_shape):]),
        #                                        dtype=y_c.dtype))

    @nn.compact
    def __call__(self, mask, metric, metric_reg, y_c):
        """
        encodes a subset of points from a dataset as a discretised functional representation

        functional representation is formed using a kernel of the form exp(norm(x_c, x_t))

        # :param x:                     context (given) datapoint locations
        :param y:                     context (given) datapoint values
        # :param x_g:                   target locations
        :param mask:                  context set mask: mask==True for datapoints in the context set, and False otherwise
        :return:                      density, y_t        if return_density==True
                                      y_t                 otherwise
                                      where     y_t:                target values
                                                density:            positive quantity representing closeness to context
                                                                    datapoints
        """

        # assert len(mask.shape) == len(x_c.shape) - 1
        # assert x_c.shape[-1] == 3, 'expecting 3d cartesian coordinates'
        # assert x_t.shape[-1] == 3, 'expecting 3d cartesian coordinates'
        # assert (len(y_c.shape) == 5 and len(x_c.shape) == 3) or (len(y_c.shape) == 4 and len(x_c.shape) == 2), \
        #     'expecting shapes (points, 3) and (points, sequence, spin, channel) or ' \
        #     '(resolution, resolution, 3) and (resolution, resolution, sequence, spin, channel)'

        share_scale_across_data_and_density = self.share_scale_across_data_and_density
        if self.return_density:
            assert share_scale_across_data_and_density is not None, 'share_psi_across_data_and_density not set'
        else:
            assert share_scale_across_data_and_density is None, 'setting share_psi_across_data_and_density ' \
                                                                'is non-sensical when density is not computed, ' \
                                                                'please set to None'
            share_scale_across_data_and_density = True  # can be True or False, set to simplest case: True

        if share_scale_across_data_and_density:
            scale_name = "softplus^{-1}(scale_4_density_and_value)"
        else:
            scale_name = "softplus^{-1}(scale_4_value)"

        K_axes, y_c_axes = ExponentialKernelSetConvolution.get_axes(mask)

        scale = nn.softplus(self.param(scale_name, self.initializer, (1,)))

        if self.small_memory_footprint:

            kernel_filter = partial(ExponentialKernelSetConvolution.small_space_complexity_kernel_filter,
                                    kernel=self.kernel, scale=scale, mask=mask, metric=metric_reg)
            value = kernel_filter(y_c=y_c)
            density = kernel_filter(y_c=jnp.ones_like(y_c))
            density = density * (density != 0) + 1 * (density == 0)
            y_t = value / density
            del value, density

            if self.return_density:

                if not self.share_scale_across_data_and_density:
                    # overwrite log_K using kernel with specific scale
                    scale = nn.softplus(self.param("softplus^{-1}(scale_4_density)", self.initializer, (1,)))

                kernel_filter = partial(ExponentialKernelSetConvolution.small_space_complexity_kernel_filter,
                                        kernel=self.kernel, scale=scale, mask=mask, metric=metric)
                density = kernel_filter(y_c=jnp.array(mask, dtype=y_c.dtype))

        else:

            k = self.kernel.k(metric_reg, scale)

            def normalise(k_):
                return ExponentialKernelSetConvolution.normalise(mask * k_, axis=K_axes)

            y_t = jnp.tensordot(normalise(k), y_c, axes=[K_axes, y_c_axes])

            if self.return_density:

                if not self.share_scale_across_data_and_density:
                    # overwrite log_K using kernel with specific scale
                    scale = nn.softplus(self.param("softplus^{-1}(scale_4_density)", self.initializer, (1,)))

                k = self.kernel.k(metric, scale)
                density = jnp.tensordot(k, mask, axes=[K_axes, y_c_axes])

        if self.return_density:
            # need to add axis placeholders or else calling nn.vmap on ExponentialKernelSetConvolution module would
            # require laborious book-keeping
            density = jax.lax.broadcast_in_dim(operand=density,
                                               shape=(*density.shape, *[1] * (len(y_t.shape) - len(density.shape))),
                                               broadcast_dimensions=tuple(range(len(density.shape))))

            return density, y_t

        else:

            return y_t


class BatchedSetConvolution(nn.Module):
    """ off-grid set convolution """
    set_convolution_module: nn.Module
    use_isotropic_scale_in_complex_plane: Optional[bool] = True
    share_scale_across_channels: Optional[bool] = True
    share_scale_across_sequence: Optional[bool] = True

    @nn.compact
    def __call__(self, mask, x_c, y_c, x_t):
        """
        :param mask:                expected to be of shape (batch, *spatial_axes)
        :param x_c:                 expected to be of shape (batch, *spatial_axes, 3)
        :param y_c:                 expected to be of shape (batch, *spatial_axes, sequence, spin, channels, complex)
        :param x_t:                 expected to be of shape (batch, *spatial_axes)
        :return:                    array of shape (batch, *spatial, sequence, spin, channels), wherein density,
                                    real value and imaginary value are stacked to form the channel dimension
        """

        assert x_c.shape[:-1] == mask.shape, 'expecting x_c with shape (*mask.shape, 3)'
        assert len(mask.shape) + len(['sequence', 'spin', 'channel', 'complex']) == len(y_c.shape), \
            'expecting y_c with shape (*mask.shape, sequence, spin, channels, complex)'
        assert y_c.shape[:-len(['sequence', 'spin', 'channel',
                                'complex'])] == mask.shape, \
            'expecting y_c with shape (*mask.shape, sequence, spin, channels, complex)'

        set_convolution_module = self.set_convolution_module

        K_axes, y_c_axes = self.set_convolution_module.func.get_axes(mask[0, ...])
        metric, metric_reg = jax.vmap(set_convolution_module().metric, in_axes=(0, 0, 0, None), out_axes=0)(mask, x_c,
                                                                                                            x_t, K_axes)

        def make_out_axes(out_axes):
            if set_convolution_module.keywords['return_density']:
                return (out_axes, out_axes)
            else:
                return out_axes

        if not self.use_isotropic_scale_in_complex_plane:
            set_convolution_module = nn.vmap(set_convolution_module,
                                             in_axes=(None, None, None, -1),
                                             out_axes=make_out_axes(-1),
                                             variable_axes={'params': -1},
                                             split_rngs={'params': True})

        if not self.share_scale_across_channels:
            set_convolution_module = nn.vmap(set_convolution_module,
                                             in_axes=(None, None, None, -2),
                                             out_axes=make_out_axes(-2),
                                             variable_axes={'params': -2},
                                             split_rngs={'params': True})

        if not self.share_scale_across_sequence:
            set_convolution_module = nn.vmap(set_convolution_module,
                                             in_axes=(None, None, None, -4),
                                             out_axes=make_out_axes(-4),
                                             variable_axes={'params': 0},
                                             split_rngs={'params': True})

        set_convolution_module = nn.vmap(set_convolution_module,
                                         in_axes=(0, 0, 0, 0),
                                         variable_axes={'params': None},
                                         split_rngs={'params': False})

        set_convolution = set_convolution_module()

        if set_convolution.return_density:

            density, value = set_convolution(mask, metric, metric_reg, y_c)

        else:

            value = set_convolution(mask, metric, metric_reg, y_c)

        ''' stacking real and imaginary values upon channel dimension '''

        value = jnp.concatenate([value[..., part] for part in range(value.shape[-1])], axis=-1)
        if set_convolution.return_density:
            density = jnp.concatenate([density[..., part] for part in range(density.shape[-1])], axis=-1)

        ''' stacking density upon channel dimension '''

        if set_convolution.return_density:

            density = jax.lax.broadcast_in_dim(density,
                                               shape=(*value.shape[:-1], density.shape[-1]),
                                               broadcast_dimensions=tuple(range(len(density.shape))))

            return jnp.concatenate([value, density], axis=-1)

        else:

            return value


class TruncateContextSetDecorator(nn.Module):
    """ truncates the context set to a pre-specified maximum number of non-masked datapoints to achieve memory
        consumption savings when calling a downstream (off-the-grid) set convolution module

        assumes datapoints are sorted, with non-masked datapoint values leading within the datapoint vectors

        such approach of (pre-emptively) sorting and statically truncating the datapoint vectors, circumvents
        difficulties when dynamically indexing GPU/jit-ed arrays, if not the outright impossibility of such indexing
    """
    set_convolution_module: nn.Module
    max_context_datapoint_count: int

    @nn.compact
    def __call__(self, mask, x_c, y_c, x_t):
        @partial(partial, tree_map)
        def trunc(z):
            return z[:, :self.max_context_datapoint_count]

        mask, x_c, y_c = trunc(mask), trunc(x_c), trunc(y_c)

        return self.set_convolution_module()(mask, x_c, y_c, x_t)


class SwapIpsiContraDecorator1(nn.Module):
    batched_set_convolution_module: Callable

    @nn.compact
    def __call__(self, mask, x_c, y_c, x_t):
        assert x_c.shape[:-1] == mask.shape, 'expecting x_c with shape (batch, *mask.shape, 3)'
        assert len(mask.shape) + len(['sequence', 'spin', 'channel', 'complex']) == len(y_c.shape), \
            'expecting y_c with shape (batch, *mask.shape, sequence, spin, channel, complex)'
        assert y_c.shape[:-len(['sequence', 'spin', 'channel', 'complex'])] == mask.shape, \
            'expecting y_c with shape (batch, *mask.shape, sequence, spin, channel, complex)'
        assert y_c.shape[-2] == 2, 'expecting channel dimension of size 2 (two ears)'

        batched_set_convolution = self.batched_set_convolution_module()

        def flip_y_axis(x):
            return x.at[..., 1].set(-x[..., 1])

        h_left_ear = batched_set_convolution(mask, x_c, y_c[..., :1, :], x_t)
        h_right_ear = batched_set_convolution(mask, x_c, y_c[..., 1:, :], flip_y_axis(x_t))

        assert h_right_ear.shape == h_left_ear.shape
        assert len(x_t.shape[1:-1]) == 2 and x_t.shape[1] == x_t.shape[2], \
            'expecting x_t to be of shape (resolution, resolution)'
        assert len(x_t.shape[:-1]) + len(['sequence', 'spin', 'channel']) == len(h_left_ear.shape), \
            'expecting h with shape (batch, resolution, resolution, sequence, spin, channel)'
        assert x_t.shape[1:-1] == h_left_ear.shape[1:3], \
            'expecting h with shape (batch, resolution, resolution, sequence, spin, channel)'

        return jnp.concatenate((h_left_ear, h_right_ear), axis=-1)


class SwapIpsiContraDecorator2(nn.Module):
    batched_set_convolution_module: Callable

    @nn.compact
    def __call__(self, mask, x_c, y_c, x_t):
        assert x_c.shape[:-1] == mask.shape, 'expecting x_c with shape (batch, *mask.shape, 3)'
        assert len(mask.shape) + len(['sequence', 'spin', 'channel', 'complex']) == len(y_c.shape), \
            'expecting y_c with shape (batch, *mask.shape, sequence, spin, channel, complex)'
        assert y_c.shape[:-len(['sequence', 'spin', 'channel', 'complex'])] == mask.shape, \
            'expecting y_c with shape (batch, *mask.shape, sequence, spin, channel, complex)'
        assert y_c.shape[-2] // 2 == y_c.shape[-2] / 2, 'expecting even number of channels'

        n_channels = y_c.shape[-2]

        def flip_y_axis(x):
            return x.at[..., 1].set(-x[..., 1])

        y_c_left_ear = y_c[..., :n_channels // 2, :]
        y_c_right_ear = y_c[..., n_channels // 2:, :]

        batched_set_convolution_module = self.batched_set_convolution_module()
        return jnp.concatenate((batched_set_convolution_module(mask, x_c, y_c_left_ear, x_t),
                                batched_set_convolution_module(mask, x_c, y_c_right_ear, flip_y_axis(x_t))),
                               axis=-1)
