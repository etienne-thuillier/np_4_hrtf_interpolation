from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional

from spin_spherical_cnns.spin_spherical_harmonics import SpinSphericalFourierTransformer
from spin_spherical_cnns.sphere_utils import ell_max_from_resolution, spin_spherical_mean
import spin_spherical_cnns.layers as ssc_layers

import models.modules.initializers as initializers


class MagnitudeNonlinearity3D(nn.Module):
    """
    Adapted from: google-research/spin_spherical_cnn/layers.py (MagnitudeNonlinearity class)
    Original Author(s): The Google Research Authors (Apache 2.0 License)
    Modifications: Adapted for use with spherical-planar signals

    Attributes
        epsilon: Small float constant to avoid division by zero.
        bias_initializer: initializer for the bias (default to zeroes).
    """
    epsilon: jnp.float32 = 1e-6
    bias_initializer: ssc_layers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        """Applies pointwise nonlinearity to 6D inputs."""
        bias = self.param("bias", self.bias_initializer,
                          (1, 1, 1, 1, inputs.shape[-2], inputs.shape[-1]))  # **** change made here

        modulus_inputs = jnp.abs(inputs)
        return (nn.relu(modulus_inputs + bias) *
                (inputs / (modulus_inputs + self.epsilon)))


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7, 8))
def swspconv_spatial_spectral(transformer, sphere_set, filter_coefficients, spins_in, spins_out, n_filter_taps,
                              spatial_resample_factor, sequence_resample_factor, sequence_filter_mode):
    """
    Adapted from: google-research/spin_spherical_cnn/layers.py (_swsconv_spatial_spectral method)
    Original Author(s): The Google Research Authors (Apache 2.0 License)
    Modifications: Added planar convolution support for time-varying spherical signals on the sphere

    Attributes
        sphere_set:             array with shape (resolution, resolution, sequence, n_spins, n_channels)
        filter_coefficients:    array with shape (ell_max + 1, spins_in, spins_out, n_channels, features, n_filter_taps)

        returned value          array with shape (resolution, resolution, n_time_samples, spins_out, features)
    """

    ''' forward sws transform, vectorized across signal taps, NB:
            + sphere_set has shape (resolution, resolution, n_time_samples, n_spins, n_channels) '''

    vmap_sequence = jax.vmap(transformer.swsft_forward_spins_channels, in_axes=(2, None), out_axes=-1)
    coefficients_in = vmap_sequence(sphere_set, spins_in)

    shape = coefficients_in.shape
    if spatial_resample_factor == 0.5:

        ''' spatial down-sampling => truncate degree (ell_max) of spherical representation '''

        assert coefficients_in.shape[0] % 2 == 0, 'case where non-even leading dimension is not coded'

        coefficients_in = coefficients_in[:shape[0] // 2, shape[0] // 2:-shape[0] // 2]

        assert filter_coefficients.shape[0] == shape[0] // 2

    else:

        assert spatial_resample_factor in [1, 2]

    ''' 
        upsampling in the sequence dimension is carried-out by imparting dilation >  1 on the input (a.k.a. signal)
        downsampling in the sequence dimension is carried-out by imparting a stride > 1 to the kernel (a.k.a. filter)
        this follows loosely:
                https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html 
    '''

    ''' default values '''

    same_length = coefficients_in.shape[-1] * sequence_resample_factor
    same_length = int(same_length)

    coefficients_in_ = coefficients_in
    filter_stride = 1
    padding_length = n_filter_taps // 2

    j0 = 0

    if sequence_resample_factor > 1:

        ''' sequence up-sampling => dilate input in sequence dimension '''

        assert n_filter_taps > sequence_resample_factor / 2, 'too short of a filter given the sequence upsampling ' \
                                                             'factor => risk of aliasing'
        assert sequence_resample_factor == int(sequence_resample_factor), 'use an integer-valued upsampling factor'
        sequence_resample_factor = int(sequence_resample_factor)

        new_shape = (*coefficients_in.shape[:-1], same_length)
        coefficients_in_ = jnp.zeros(new_shape, dtype=coefficients_in.dtype)

        i0 = (same_length // 2) % sequence_resample_factor
        coefficients_in_ = coefficients_in_.at[..., i0::sequence_resample_factor].set(coefficients_in)
        del i0

    elif sequence_resample_factor < 1:

        ''' sequence down-sampling => use larger stride than 1 '''

        filter_stride = int(1.0 / sequence_resample_factor)
        assert coefficients_in.shape[-1] / filter_stride == same_length, 'being conservative here:' \
                                                                         'probably best to use ' \
                                                                         'downsample factors that are ' \
                                                                         'perfect divisor of sequence length'
        valid_length = same_length - (n_filter_taps - 1)
        valid_center = padding_length + valid_length // 2
        j0 = valid_center % filter_stride

    ''' convolution accross time (direct form) and in frequecy (in spherical harmonics domain) '''

    mirror_frequency_axis = True  # **** make this a user-adjustable parameter

    if mirror_frequency_axis and padding_length > 0:

        pad_low_frequency = coefficients_in_[..., 1:padding_length + 1]
        pad_low_frequency = jnp.flip(pad_low_frequency, axis=-1)
        pad_low_frequency = jnp.conj(pad_low_frequency)

        pad_high_frequency = coefficients_in_[..., -padding_length:]
        pad_high_frequency = jnp.flip(pad_high_frequency, axis=-1)
        pad_high_frequency = jnp.conj(pad_high_frequency)

        nyquist = jnp.zeros((*coefficients_in_.shape[:-1], 1), dtype=coefficients_in_.dtype)
        pad_high_frequency = jnp.concatenate((nyquist, pad_high_frequency[..., 1:]), axis=-1)

    else:

        pad_low_frequency = jnp.zeros(shape=(*coefficients_in_.shape[:-1], padding_length),
                                      dtype=coefficients_in_.dtype)
        pad_high_frequency = pad_low_frequency

    padded_coefficients_in = jnp.concatenate((pad_low_frequency, coefficients_in_, pad_high_frequency), axis=-1)

    def body_fun(i, val):

        update = jax.vmap(jnp.einsum, in_axes=(None, -1, None), out_axes=-1)("lmic,liocd->lmod",
                                                                             padded_coefficients_in,
                                                                             filter_coefficients[..., i])
        update = jnp.roll(update, shift=-i, axis=-1)

        '''
            the following is admittedly wasteful, an alternative would be to cycle through the input sequence samples
            instead of the filter taps, using a rolling buffer with a step equal to the filter stride

            this would have the benefit of lesser memory consumption, requiring only n_filter_taps sh representation of
            the input an any time, the spherical harmonic transform could then be integrated to the loop
        '''
        update = update[..., j0::filter_stride]

        return val + update

    features_out = filter_coefficients.shape[-2]
    shape = (
        *padded_coefficients_in.shape[:2], len(spins_out), features_out,
        padded_coefficients_in.shape[-1] // filter_stride)
    coefficients_out = jax.lax.fori_loop(lower=0,
                                         upper=n_filter_taps,
                                         body_fun=body_fun,
                                         init_val=jnp.zeros(shape=shape, dtype=filter_coefficients.dtype))

    ''' removing non-valid part of time convolution result '''
    if n_filter_taps > 1:  # since, should we apply truncation below when n_filter_taps == 1, we would en up with
        # coefficients_out[..., :0] which is an empty vector...

        if sequence_filter_mode == 'same':

            coefficients_out = coefficients_out[..., :same_length]

        elif sequence_filter_mode == 'full':

            coefficients_out = jnp.roll(coefficients_out, shift=padding_length, axis=-1)[...,
                               :same_length + (n_filter_taps - 1)]

        elif sequence_filter_mode == 'valid':

            coefficients_out = coefficients_out[..., padding_length:padding_length + same_length - (n_filter_taps - 1)]

        else:

            assert False, 'non-supported mode'

    ''' NB: shape of coefficients_out is now (ell_max + 1, 2*ell_max + 1(?), spins_out, features, n_time_samples) '''

    ''' up-sampling the case being'''

    if spatial_resample_factor == 2:
        ''' spatial up-sampling => pad spherical representation with zeros to increase spherical representation's
            degree (ell_max) '''

        shape = coefficients_out.shape
        zeros_shape = (shape[0], (shape[1] + 1) // 2, *shape[2:])
        zeros = jnp.zeros(zeros_shape, dtype=coefficients_out.dtype)
        coefficients_out = jnp.concatenate((zeros, coefficients_out, zeros), axis=1)

        coefficients_out = jnp.concatenate((coefficients_out, jnp.zeros_like(coefficients_out)), axis=0)

    ''' backward sws transform (vmapped across time samples) '''

    sphere_set_out = jax.vmap(transformer.swsft_backward_spins_channels,
                              in_axes=(-1, None),
                              out_axes=-1)(coefficients_out, spins_out)

    ''' putting time samples dimension back to third index '''

    return sphere_set_out.transpose(0, 1, 4, 2, 3)


class SpinSphericalPlanarConvolution(nn.Module):
    """
    Adapted from: google-research/spin_spherical_cnn/layers.py (SpinSphericalConvolution class)
    Original Author(s): The Google Research Authors (Apache 2.0 License)
    Modifications: Added planar convolution support for time-varying spherical signals on the sphere

    See original license at: https://www.apache.org/licenses/LICENSE-2.0

    A spherical-planar variant to the spin-weighted spherical convolutional layer, which adds an additional (e.g. time,
    frequency, ...) dimension to the features that can be filtered. This allows for example convolution time-varying
    spherical signals. For clarity, we designate additional dimension as "time" below, but this is of course not meant
    to be limitative as the layer applies for spherical signals of frequency-domain quantities, temperature...

    Wraps swsconv_spatial_spectral_3D(), initializing and keeping track of the learnable filter.

    Attributes
      features:                 int, number of output features (channels).
      spins_in:                 (n_spins_in,) Sequence of int containing the input spins.
      spins_out:                (n_spins_out,) Sequence of int containing the output spins.
      transformer:              SpinSphericalFourierTransformer instance.
      num_filter_params:        Number of parameters per filter. Fewer parameters results
                                in more localized filters.
      initializer:              initializer for the filter spectrum.
    """
    features: int
    spins_in: Sequence[int]
    spins_out: Sequence[int]
    transformer: SpinSphericalFourierTransformer
    num_filter_params: int  # Optional[int] = None
    num_filter_taps: int  # Optional[int] = 3
    initializer: ssc_layers.Initializer = ssc_layers.default_initializer
    spatial_resample_factor: Optional[float] = 1.0
    sequence_resample_factor: Optional[float] = 1.0
    sequence_filter_mode: Optional[str] = 'same'
    is_depthwise: Optional[bool] = False

    def _get_kernel(self, ell_max, num_channels_in):
        assert False, 'not adapted yet: this is the original code from SpinSphericalConvolution'
        kernel_shape = (ell_max + 1, len(self.spins_in), len(self.spins_out),
                        num_channels_in, self.features)
        return self.param("kernel", self.initializer, kernel_shape)

    def _get_localized_kernel(self, ell_max, num_channels_in):
        # We interpolate along ell to obtain all weights from the learnable weights,
        # hence it doesn't make sense to have more parameters than num_ell.
        if self.num_filter_params > ell_max + 1:
            raise ValueError("num_filter_params must be <= ell_max + 1")
        ell_in = jnp.linspace(0, 1, self.num_filter_params)
        ell_out = jnp.linspace(0, 1, ell_max + 1)
        # `vectorize` is over leading dimensions, so we put ell as the last
        # dimension and transpose it to the first later.
        # ****

        if self.is_depthwise:

            learnable_shape = (len(self.spins_in), len(self.spins_out),
                               num_channels_in, 1,
                               self.num_filter_taps,
                               self.num_filter_params)
        else:

            learnable_shape = (len(self.spins_in), len(self.spins_out),
                               num_channels_in, self.features,
                               self.num_filter_taps,  # ****
                               self.num_filter_params)
        # ****
        learnable_weights = self.param("kernel", self.initializer, learnable_shape)
        # `jnp.interp` works on 1D inputs; we vectorize it to interpolate over a
        # single dimension of n-D inputs.
        vectorized_interp = jnp.vectorize(jnp.interp, signature="(m),(n),(n)->(m)")
        weights = vectorized_interp(ell_out, ell_in, learnable_weights)
        # Make ell the first dimension.
        return weights.transpose((5, 0, 1, 2, 3, 4))  # ****

    @nn.compact
    def __call__(self, sphere_set):
        """Applies convolution to inputs.

        Args:
          sphere_set: A (batch_size, resolution, resolution, n_time_samples_in, n_spins_in,
            n_channels_in) array of spin-weighted spherical functions (SWSF) with
            equiangular sampling.

        Returns:
          A (batch_size, resolution, resolution, n_time_samples_out, n_spins_out, n_channels_out)
          complex64 array of SWSF with equiangular H&W sampling.
        """
        resolution = sphere_set.shape[1]
        if sphere_set.shape[2] != resolution:
            raise ValueError("Axes 1 and 2 must have the same dimensions!")
        if sphere_set.shape[4] != len(list(self.spins_in)):  # ****
            raise ValueError("Input axis 4 (spins_in) doesn't match layer's.")  # ****

        # Make sure constants contain all spins for input resolution.
        for spin in set(self.spins_in).union(self.spins_out):
            if not self.transformer.validate(resolution, spin):
                raise ValueError("Constants are invalid for given input!")

        if self.spatial_resample_factor == 0.5:

            ''' down-sampling is carried-out by pre-emptively pruning higher degree components of the input signal's
                spherical harmonic spectrum 
                => filter's ell_max determined by output (resampled) resolution
            '''

            ell_max = ell_max_from_resolution(resolution // 2)

        elif self.spatial_resample_factor == 2.0:

            ''' up-sampling is carried-out by padding zeros into the output signal's spherical harmonic spectrum, so
                as to achieve the required resolution, the effective filtering being carried-out on the non-resampled
                input signal
                => filter's ell_max determined by input (non-resampled) resolution
            '''

            ell_max = ell_max_from_resolution(resolution)

        else:

            assert self.spatial_resample_factor == 1.0
            ell_max = ell_max_from_resolution(resolution)

        num_channels_in = sphere_set.shape[-1]

        if self.is_depthwise:
            assert self.features == num_channels_in, 'expecting equal number of features and channels in depthwise' \
                                                     'variant, i.e. each output feature is computed exclusively from a' \
                                                     'unique input channel using a corresponding specific filter'
        if self.num_filter_params is None:

            kernel = self._get_kernel(ell_max, num_channels_in)

        else:

            kernel = self._get_localized_kernel(ell_max, num_channels_in)

        ''' vmap across batch dimension'''

        conv = jax.vmap(swspconv_spatial_spectral, in_axes=(None, 0, None, None, None, None, None, None, None),
                        out_axes=0)

        if self.is_depthwise:

            sphere_set = jnp.expand_dims(sphere_set, axis=-1)
            kernel = jnp.expand_dims(kernel, axis=-3)

            ''' vmap across channel dimension: each input channel will be filtered to produce one feature output  '''

            # sphere_set.shape = (batch, resolution, resolution, sequence, n_spins, n_channels, 1)
            # kernel.shape = (ell_max + 1, spins_in, spins_out, n_channels, 1, features=1, n_filter_taps)
            conv = jax.vmap(conv, in_axes=(None, -2, -4, None, None, None, None, None, None), out_axes=-1)

            return conv(self.transformer,
                        sphere_set,
                        kernel,
                        self.spins_in, self.spins_out,
                        self.num_filter_taps,
                        self.spatial_resample_factor,
                        self.sequence_resample_factor,
                        self.sequence_filter_mode)[..., 0]

        else:

            return conv(self.transformer,
                        sphere_set,
                        kernel,
                        self.spins_in, self.spins_out,
                        self.num_filter_taps,
                        self.spatial_resample_factor,
                        self.sequence_resample_factor,
                        self.sequence_filter_mode)


class SphericalPlanarBatchNormalization(nn.Module):
    """
    Adapted from: google-research/spin_spherical_cnn/layers.py (SphericalBatchNormalization class)
    Original Author(s): The Google Research Authors (Apache 2.0 License)
    Modifications: Adapted for use with spherical-planar signals

    See original license at: https://www.apache.org/licenses/LICENSE-2.0

    A 3D variant to the 2D spin-weighted spherical batch norm layer, whic adds an additional (e.g. time, frequency,
    ...) dimension to the features that can be filtered. This allows for example convolution time-varying spherical
    signals. For clarity, we designate additional dimension as "time" below, but this is of course not meant to be
    limitative as the layer applies for spherical signals of frequency-domain quantities, temperature...

    Attributes
        use_running_stats:  if True, the statistics stored in batch_stats
                            will be used instead of computing the batch statistics on the input.
        momentum:           decay rate for the exponential moving average of
                            the batch statistics.
        centered:           When False, skips mean-subtraction step.
        epsilon:            a small float added to variance to avoid dividing by zero.
        use_bias:           if True, add a complex-valued learned bias.
        use_scale:          if True, multiply by a complex-valued learned scale.
        bias_init:          initializer for bias, by default, zero.
        scale_init:         initializer for scale, by default, one.
        axis_name:          the axis name used to combine batch statistics from multiple
                            devices. See `jax.pmap` for a description of axis names (default: None).
    """
    use_running_stats: Optional[bool] = None
    momentum: float = 0.99
    epsilon: float = 1e-5
    centered: bool = True
    use_bias: bool = True
    use_scale: bool = True
    bias_init: ssc_layers.Initializer = initializers._complex_zeros_initializer
    scale_init: ssc_layers.Initializer = initializers._complex_ones_initializer
    axis_name: Optional[str] = None

    @nn.compact
    def __call__(self,
                 inputs,
                 use_running_stats=None,
                 weights=None):
        """Normalizes the input using batch (optional) means and variances.

		Stats are computed over the batch and spherical dimensions: (0, 1, 2).

		Args:
		  inputs: An array of dimensions (batch_size, resolution, resolution,
			n_spins_in, n_channels_in).
		  use_running_stats: if true, the statistics stored in batch_stats will be
			used instead of computing the batch statistics on the input.
		  weights: An array of dimensions (batch_size,) assigning weights for
			each batch element. Useful for masking.

		Returns:
		  Normalized inputs (the same shape as inputs).
		"""
        use_running_stats = nn.module.merge_param(
            "use_running_stats", self.use_running_stats, use_running_stats)

        # Normalization is independent per spin per channel.
        num_spins, num_channels = inputs.shape[-2:]
        feature_shape = (1, 1, 1, 1, num_spins, num_channels)  # **** change occured here
        reduced_feature_shape = (num_spins, num_channels)

        initializing = not self.has_variable("batch_stats", "variance")

        running_variance = self.variable("batch_stats", "variance",
                                         lambda s: jnp.ones(s, jnp.float32),
                                         reduced_feature_shape)

        if self.centered:
            running_mean = self.variable("batch_stats", "mean",
                                         lambda s: jnp.zeros(s, jnp.complex64),
                                         reduced_feature_shape)

        if use_running_stats:
            variance = running_variance.value
            if self.centered:
                mean = running_mean.value
        else:
            # Compute the spherical mean over the spherical grid dimensions, then a
            # conventional mean over the batch.
            if self.centered:
                mean = jnp.average(inputs, axis=3)
                mean = spin_spherical_mean(mean)
                mean = jnp.average(mean, axis=0, weights=weights)
            # Complex variance is E[x x*] - E[x]E[x*].
            # For spin != 0, E[x] should be zero, although due to discretization this
            # is not always true. We only use E[x x*] here.
            # E[x x*]:
            mean_abs_squared = jnp.average(inputs * inputs.conj(), axis=3)
            mean_abs_squared = spin_spherical_mean(mean_abs_squared)
            mean_abs_squared = jnp.average(mean_abs_squared, axis=0, weights=weights)
            # Aggregate means over devices.
            if self.axis_name is not None and not initializing:
                if self.centered:
                    mean = lax.pmean(mean, axis_name=self.axis_name)
                mean_abs_squared = lax.pmean(mean_abs_squared, axis_name=self.axis_name)

            # Imaginary part is negligible.
            variance = mean_abs_squared.real

            if not initializing:
                running_variance.value = (self.momentum * running_variance.value +
                                          (1 - self.momentum) * variance)
                if self.centered:
                    running_mean.value = (self.momentum * running_mean.value +
                                          (1 - self.momentum) * mean)

        if self.centered:
            outputs = inputs - mean.reshape(feature_shape)
        else:
            outputs = inputs

        factor = lax.rsqrt(variance.reshape(feature_shape) + self.epsilon)
        if self.use_scale:
            scale = self.param("scale",
                               self.scale_init,
                               reduced_feature_shape).reshape(feature_shape)
            factor = factor * scale

        outputs = outputs * factor

        if self.use_bias:
            bias = self.param("bias",
                              self.bias_init,
                              reduced_feature_shape).reshape(feature_shape)
            outputs = outputs + bias

        return outputs


class SpinSphericalPlanarMagnitudeNonlin(nn.Module):
    """ version of SpinSphericalPlanarBatchNormMagnitudeNonlin without batch normalisation """
    spins: Sequence[int]
    bias_initializer: ssc_layers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        """ Calls appropriate nonlinearity per spin."""

        outputs = []
        for i, spin in enumerate(self.spins):

            inputs_spin = inputs[Ellipsis, [i], :]

            if spin == 0:

                outputs_spin = nn.leaky_relu(inputs_spin.real)

            else:

                outputs_spin = MagnitudeNonlinearity3D(bias_initializer=self.bias_initializer,
                                                       name=f"magnitude_nonlin_{i}")(inputs_spin)
            outputs.append(outputs_spin)

        return jnp.concatenate(outputs, axis=-2)


class SpinSphericalPlanarBatchNormMagnitudeNonlin(nn.Module):
    """
    Adapted from: google-research/spin_spherical_cnn/layers.py (SpinSphericalBatchNormalizationNonlinearity class)
    Original Author(s): The Google Research Authors (Apache 2.0 License)
    Modifications: Adapted for use with spherical-planar signals

    Attributes
        spins:                  (n_spins,) Sequence of int containing the input spins.
        use_running_stats:      if True, the statistics stored in batch_stats
                                will be used instead of computing the batch statistics on the input.
        momentum:               decay rate for the exponential moving average of the batch statistics.
        epsilon:                a small float added to variance to avoid dividing by zero.
        axis_name:              the axis name used to combine batch statistics from multiple devices.
                                See `jax.pmap` for a description of axis names (default: None).
        bias_initializer:       initializer for MagnitudeNonlinearity bias, by default, zero.
    """
    spins: Sequence[int]
    use_running_stats: Optional[bool] = None
    momentum: float = 0.99
    epsilon: float = 1e-5
    axis_name: Optional[str] = None
    bias_initializer: ssc_layers.Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self,
                 inputs,
                 use_running_stats=None,
                 weights=None):
        """Calls appropriate batch normalization and nonlinearity per spin."""
        use_running_stats = nn.module.merge_param(
            "use_running_stats", self.use_running_stats, use_running_stats)

        options = dict(use_running_stats=use_running_stats,
                       momentum=self.momentum,
                       epsilon=self.epsilon,
                       axis_name=self.axis_name)
        outputs = []
        for i, spin in enumerate(self.spins):
            inputs_spin = inputs[Ellipsis, [i], :]
            if spin == 0:
                outputs_spin = SphericalBatchNormalization3D(use_bias=True,
                                                             centered=True,
                                                             **options)(inputs_spin,
                                                                        weights=weights)
                outputs_spin = nn.leaky_relu(outputs_spin.real)
            else:
                outputs_spin = SphericalBatchNormalization3D(use_bias=False,
                                                             centered=False,
                                                             **options)(inputs_spin,
                                                                        weights=weights)
                outputs_spin = MagnitudeNonlinearity3D(bias_initializer=self.bias_initializer,
                                                       name=f"magnitude_nonlin_{i}")(outputs_spin)
            outputs.append(outputs_spin)

        return jnp.concatenate(outputs, axis=-2)
