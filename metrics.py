from functools import partial

import jax.numpy as jnp
import jax.scipy
from jax import vmap
from jax.tree_util import tree_map

from utilities.utilities import collapse_complex_axis, expand_complex_axis


def log_likelihood(y, mu, sigma):
    """
        log-likelihood under normal probability density function

        provided sigma is assumed > 0
    """
    assert jnp.isrealobj(y) and jnp.isrealobj(mu) and jnp.isrealobj(sigma)

    return - (y - mu) ** 2 / (2 * sigma ** 2) - jnp.log(sigma) - jnp.log(jnp.sqrt(2 * jnp.pi))


def per_feature_negative_log_likelihood(y, mu, sigma):
    f = vmap(log_likelihood, in_axes=(0, 0, 0))  # point-wise values over batch
    f = vmap(f, in_axes=(None, 0, 0))  # point-wise values over samples

    if sigma.shape[-1] == 1 and mu.shape[-1] == 2:
        sigma = sigma[..., 0]
        f = vmap(f, in_axes=(-1, -1, None), out_axes=-1)  # point-wise values over complex dimension

    return -f(y, mu, sigma)


def per_feature_relative_error_db(y, mu, sigma=None, reduce_functor=lambda z: z):
    def power(z):
        return (z * jnp.conj(z)).real

    def squared_error(z_hat, z):
        return power(z_hat - z)

    e2 = squared_error(collapse_complex_axis(mu), collapse_complex_axis(y))
    e2 = expand_complex_axis(e2)

    p = power(collapse_complex_axis(y))
    p = expand_complex_axis(p)

    eps = jnp.finfo(p).eps

    return 10.0 * (jnp.log10(reduce_functor(e2) + eps) - jnp.log10(reduce_functor(p) + eps))


def per_location_log_spectral_distortion_db(y, mu, sigma, sequence_axis=3, ear_axis=5):
    """
    log-spectral distortion, e.g. according to (9) in

    Ito - 2022 -    HEAD-RELATED TRANSFER FUNCTION INTERPOLATION FROM SPATIALLY SPARSE MEASUREMENTS USING AUTOENCODER
                    WITH SOURCE POSITION CONDITIONING
    """

    assert y.shape == mu.shape[1:]

    def power(z):
        return (z * jnp.conj(z)).real

    p_hat = power(collapse_complex_axis(mu))
    p_hat = expand_complex_axis(p_hat)

    p = power(collapse_complex_axis(y))
    p = expand_complex_axis(p)

    eps = jnp.finfo(p).eps
    metric = (10.0 * (jnp.log10(p_hat + eps) - jnp.log10(p + eps))) ** 2

    # mean over sequence (i.e. frequency)
    metric = jnp.mean(metric, axis=sequence_axis, keepdims=True)

    metric = jnp.sqrt(metric)

    # mean over ears
    assert metric.shape[ear_axis] == 2
    metric = jnp.mean(metric, axis=ear_axis, keepdims=True)

    return metric


def per_feature_log_mag_distance_db(y, mu, sigma=None, reduce_functor=lambda z: z):
    def power(z):
        return (z * jnp.conj(z)).real

    p_hat = power(collapse_complex_axis(mu))
    p_hat = expand_complex_axis(p_hat)

    p = power(collapse_complex_axis(y))
    p = expand_complex_axis(p)

    eps = jnp.finfo(p).eps

    return 10.0 * jnp.abs(jnp.log10(reduce_functor(p_hat) + eps) - jnp.log10(reduce_functor(p) + eps))


def weighted_feature_metric(w, mask, y, mu, sigma, per_feature_metric, weighting_function=None,
                            preprocessing_transform=None):
    weighted_feature_metric_sanity_check(y, mu, sigma)

    if preprocessing_transform is not None:
        y, mu, sigma = preprocessing_transform(y, mu, sigma)

    m = tree_map(per_feature_metric, y, mu, sigma)

    if weighting_function is not None:
        weights = weighting_function(y)
        m = tree_map(lambda z1, z2: z1 * z2, m, weights)

    return m


def reduced_per_feature_metric(w, mask, y, mu, sigma, per_feature_metric, exclude_context_points,
                               sample_reduction_mode, weighting_function=None, preprocessing_transform=None,
                               reduction_mode='mean'):
    """ returns a reduced (scalar value) for each entry in pytree y, corresponding to the per-subject average or sum
        (depending on setup) of the values returned by the specified per_feature_metric function

        NB: assumes that leading dimension is sample, i.e. shape = (sample, s^2 location, ...)
    """

    if exclude_context_points:
        w = w * ~mask

    metric = weighted_feature_metric(w=w, mask=mask, y=y, mu=mu, sigma=sigma,
                                     per_feature_metric=per_feature_metric,
                                     weighting_function=weighting_function,
                                     preprocessing_transform=preprocessing_transform)

    reduce_feature_dimensions = partial(reduce_pointwise_metric_values,
                                        dimensions=('space', 'sequence', 'spin', 'channel', 'complex'),
                                        spatial_weights=w, keepdims=False, reduction_mode=reduction_mode)
    metric = reduce_feature_dimensions(metric)

    @partial(partial, tree_map)
    def reduce_sample_dimension(z):

        if sample_reduction_mode == 'mean':
            return z.mean(axis=0)

        if sample_reduction_mode == 'sum':

            return z.sum(axis=0)

        elif sample_reduction_mode == 'Monte-Carlo: log-likelihood':
            # L = metric.shape[0] (the # of Monte-Carlo samples for each batch elements)
            # L > 1: convlnp case -> we want to reduce the Monte-Carlo samples using logsumexp (on corresp. axis), i.e.
            # 		logsumexp(metric, axis=0) - log(metric.shape[0])
            # ... where log(L) ensures the metrics are comparable across models trained with differing number of
            #     Monte-Carlo samples L
            # L == 1: convcnp case -> we simply want get rid of singleton leading dimension, i.e.
            # 	  metric <- metric[0]
            # ... but note that in this case (L == 1):
            # 		logsumexp(metric, axis=0) - log(metric.shape[0]) = logsumexp(metric, axis=0) - log(1)
            #											   			 = logsumexp(metric, axis=0)
            #											   			 = metric[0]
            # ... hence the following can be safely used for both cases:
            return jax.scipy.special.logsumexp(z, axis=0) - jnp.log(z.shape[0])

        elif sample_reduction_mode == 'Monte-Carlo: negative log-likelihood':

            # see discussion for 'Monte-Carlo: log-likelihood' above
            z = -z
            z = jax.scipy.special.logsumexp(z, axis=0) - jnp.log(z.shape[0])
            return -z

        else:

            raise ValueError(
                'allowed modes for sample_reduction_mode in {''mean'', ''sum'', ''Monte-Carlo: log-likelihood'', ''Monte-Carlo: negative log-likelihood''}')

    return reduce_sample_dimension(metric)


@partial(jax.jit, static_argnums=(3, 4, 5, 6), device=jax.devices("cpu")[0])
def mv_mse(y, mu, sigma, n_bins, return_std_dev=False, transform_error_squared=lambda pow: pow,
           transform_variance=lambda pow: pow):
    """
        utility for computing rmv and rmse computed according to Equations (6) and (7) from

            'Levi - 2022 - Evaluating and Calibrating Uncertainty Prediction in Regression Tasks'

        returns mean variance (mv) and mean squared error (mse) such that rmv and rmse can be computed by simple
        application of square root

        optionally returns standard deviation of squared error, i.e. sse
    """

    assert ~jnp.iscomplexobj(y) and ~jnp.iscomplexobj(mu)

    assert len(y.shape) == 1
    assert len(mu.shape) == 1
    assert len(sigma.shape) == 1

    assert mu.shape[0] == y.shape[0]
    assert sigma.shape[0] == y.shape[0]

    n_datapoints_per_bin = y.shape[0] // n_bins

    sorting_indices = jnp.argsort(sigma, axis=0)
    error = y - mu

    mv = list()
    mse = list()
    sse = list()
    for bin in range(n_bins):

        bin_indices = sorting_indices[bin * n_datapoints_per_bin:(bin + 1) * n_datapoints_per_bin]

        mv += [transform_variance(sigma[bin_indices] ** 2).mean()]

        mse += [transform_error_squared(error[bin_indices] ** 2).mean()]
        # mse += [(e * jnp.conj(e)).real.mean()]
        # mse += [(e * jnp.conj(e)).real.mean()]

        if return_std_dev:
            sse += [transform_error_squared(error[bin_indices] ** 2).std()]

    if return_std_dev:
        return jnp.array(mv), jnp.array(mse), jnp.array(sse)
    else:
        return jnp.array(mv), jnp.array(mse)


''' helpers '''


# TODO: remove tree_map in helpers below, as to make clearer that these functions transform arrays

def reduce_pointwise_metric_values(metric, dimensions=('space', 'sequence', 'spin', 'channel', 'complex'),
                                   spatial_weights=None, keepdims=False, reduction_mode='mean'):
    """ reduces pointwise metric values along specified dimensions """

    if reduction_mode == 'mean':
        reduce = jnp.mean
    else:
        assert reduction_mode == 'sum'
        reduce = jnp.sum

    def reduce_on_grid_values(z):

        assert len(
            z.shape) == 8, 'expecting shape (sample, resolution, resolution, points, sequence, n_spins, n_channels, complex)'

        if 'space' in dimensions:

            if z.shape[2] == 1 and z.shape[3] == 1:  # dimension already reduced

                if not keepdims:
                    z = z[:, :, 0, 0, ...]

            else:

                assert spatial_weights is not None
                assert len(spatial_weights.shape) == 3, 'expecting shape (batch_size, resolution, resolution)'

                w = spatial_weights

                if reduction_mode == 'mean':
                    w /= spatial_weights.sum((-2, -1), keepdims=True)

                z = jnp.einsum('sbpqwxyz,bpq->sbwxyz', z, w)

                if keepdims:
                    z = jnp.expand_dims(z, axis=2)
                    z = jnp.expand_dims(z, axis=2)

        if 'sequence' in dimensions:
            z = reduce(z, axis=-4, keepdims=keepdims)

        if 'spin' in dimensions:
            z = reduce(z, axis=-3, keepdims=keepdims)

        if 'channel' in dimensions:
            z = reduce(z, axis=-2, keepdims=keepdims)

        if 'complex' in dimensions:
            z = reduce(z, axis=-1, keepdims=keepdims)

        return z

    def reduce_off_grid_values(z):

        assert len(z.shape) == 7, 'expecting shape (sample, batch_size, points, sequence, n_spins, n_channels, complex)'

        if 'space' in dimensions:

            if z.shape[2] == 1:  # dimension already reduced

                if not keepdims:
                    z = z[:, :, 0, ...]

            else:

                assert spatial_weights is not None
                assert len(spatial_weights.shape) == 2, 'expecting shape (batch_size, points)'

                w = spatial_weights

                if reduction_mode == 'mean':
                    w /= spatial_weights.sum(-1, keepdims=True)

                z = jnp.einsum('sbpwxyz,bp->sbwxyz', z, w)

                if keepdims:
                    z = jnp.expand_dims(z, axis=2)

        if 'sequence' in dimensions:
            z = reduce(z, axis=-4, keepdims=keepdims)

        if 'spin' in dimensions:
            z = reduce(z, axis=-3, keepdims=keepdims)

        if 'channel' in dimensions:
            z = reduce(z, axis=-2, keepdims=keepdims)

        if 'complex' in dimensions:
            z = reduce(z, axis=-1, keepdims=keepdims)

        return z

    @partial(partial, tree_map)
    def multimap_reduce(z):

        if len(z.shape) == 8:

            return reduce_on_grid_values(z)

        else:

            assert len(z.shape) == 7

            return reduce_off_grid_values(z)

    return multimap_reduce(metric)


''' sanity checks '''


def weighted_feature_metric_sanity_check(y, mu, sigma):
    @partial(partial, tree_map)
    def check(y_, mu_, sigma_):
        def check_mu_sigma(z):
            assert len(z.shape) in [7, 8], 'expecting array of shape (sample, batch, resolution, resolution, sequence, ' \
                                           'spin, channel, complex) or (sample, batch, points, sequence, spin, channel, ' \
                                           'complex)'

        check_mu_sigma(mu_)
        check_mu_sigma(sigma_)
        assert len(y_.shape) in [6, 7], 'expecting array of shape (batch, resolution, resolution, sequence, spin, ' \
                                        'channel, complex) or (batch, points, sequence, spin, channel, complex)'
        assert y_.shape[-2] == sigma_.shape[-2], 'channel dimension must be equal'

    check(y, mu, sigma)
