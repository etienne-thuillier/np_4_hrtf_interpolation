from functools import partial
import typing
import flax.linen as nn
import jax
from jax.tree_util import tree_map
import jax.numpy as jnp

from utilities.utilities import hermitian_pad, truncate_hermitian_pad


class ModelDecorator(nn.Module):
    model_module: nn.Module
    input_preprocessing_mode: str
    noise_mode: str
    hermitian_padding: typing.Optional[int] = 0

    @staticmethod
    def sigma_transform_vmap(sigma, mu_data, sigma_floor):

        assert False, 'deprecated'

        if sigma.shape[-1] == 2:                                # non-isotropic complex noise model
            f = jax.vmap(ModelDecorator.sigma_transform,
                         in_axes=(-1, -1, None),#{'absolute': None, 'relative': None}),
                         out_axes=-1)
        elif sigma.shape[-1] == 1:                              # isotropic complex noise model
            f = ModelDecorator.sigma_transform
        else:
            raise ValueError
        return f(sigma, mu_data, sigma_floor)

    @staticmethod
    def sigma_transform(sigma, mu_data, sigma_floor):

        assert 'absolute' in sigma_floor.keys()
        assert 'relative' in sigma_floor.keys()

        if sigma_floor['absolute'] is not None and sigma_floor['relative'] is not None:

            assert sigma_floor['absolute'] > 0, 'positive (linear) value expected, something is fishy here'
            assert sigma_floor['relative'] < 0, 'negative dB value expected, something is fishy here'

            def power(mu):
                assert jnp.isrealobj(mu)
                return (mu ** 2).sum(-1, keepdims=True)

            sigma_floor_relative = jnp.sqrt(power(mu_data) * 10 ** (sigma_floor['relative']/20))
            sigma_floor = (sigma_floor_relative < sigma_floor['absolute']) * sigma_floor['absolute'] + \
                          (sigma_floor_relative >= sigma_floor['absolute']) * sigma_floor_relative

            return sigma_floor + nn.softplus(sigma - sigma_floor)

        elif sigma_floor['absolute'] is not None:

            assert sigma_floor['absolute'] > 0, 'positive (linear) value expected, something is fishy here'
            return sigma_floor['absolute'] + (1 - sigma_floor['absolute']) * nn.softplus(sigma)

        elif sigma_floor['absolute'] is None and sigma_floor['relative'] is None:

            return sigma

        else:

            raise ValueError

    def post_process_sigma(self, sigma):

        if self.noise_mode == 'homoskedastic':

            @partial(partial, tree_map)
            def mean_over_space_dimension(s):
                return jax.lax.broadcast_in_dim(jnp.mean(s, axis=2, keepdims=True),
                                                shape=s.shape,
                                                broadcast_dimensions=list(range(len(s.shape))))

            sigma = mean_over_space_dimension(sigma)

        elif 'fixed' in self.noise_mode:

            fixed_sigma_value = float(self.noise_mode.split('sigma=')[-1])
            sigma = tree_map(lambda s: fixed_sigma_value * jnp.ones_like(s), sigma)

        else:

            assert self.noise_mode == 'heteroskedastic'
        # do nothing in this case: use models' predictions of sigma

        return sigma

    def center(self, x_c, y_c, x_t, mask, rng, L, mu_data, sigma_data):

        y_c = tree_map(lambda z, mean: z - mean, y_c, mu_data)

        mu, sigma = self.model_module()(x_c, y_c, x_t, mask, rng, L)
        sigma = self.post_process_sigma(sigma)

        mu = tree_map(lambda y, mean: y + mean, mu, mu_data)

        # sigma = tree_map(ModelDecorator.sigma_transform_vmap, sigma, mu_data, self.sigma_floor)

        return mu, sigma

    def normalised(self, x_c, y_c, x_t, mask, rng, L, mu_data, sigma_data):

        assert False, 'deprecated'

        y_c = tree_map(lambda z, mean, std: (z - mean) / std,
                           y_c,
                           mu_data,
                           sigma_data)

        mu, sigma = self.model_module()(x_c, y_c, x_t, mask, rng, L)
        sigma = self.post_process_sigma(sigma)

        mu = tree_map(lambda y, mean, std: y * std + mean, mu, mu_data, sigma_data)

        # TODO: somewhat hackish way of putting a floor to predicted sigma, although this has also been done in
        #       model... this prevents un-whitening from producing 0-valued sigma, which tend to occur in non-isotropic
        #       case. the alternative is to drop the model decorator and to integrate the whitening within the model
        #       itself, e.g. within the post-processing step
        # **** where is the floor??
        sigma = tree_map(lambda z, std: z * std, sigma, sigma_data)

        sigma = tree_map(ModelDecorator.sigma_transform_vmap, sigma, mu_data, self.sigma_floor.unfreeze())

        return mu, sigma

    @nn.compact
    def __call__(self, x_c, y_c, x_t, mask, rng, L, mu_data, sigma_data):

        if self.hermitian_padding != 0:
            assert not isinstance(y_c, dict), 'case not yest supported'

            y_c = hermitian_pad(y_c, padding=self.hermitian_padding)
            mu_data = hermitian_pad(mu_data, padding=self.hermitian_padding)
            sigma_data = hermitian_pad(sigma_data, padding=self.hermitian_padding)

        if self.input_preprocessing_mode == 'centered':

            mu, sigma = self.center(x_c, y_c, x_t, mask, rng, L, mu_data, sigma_data)

        elif self.input_preprocessing_mode == 'normalised':

            assert False, 'deprecated'
            mu, sigma = self.normalised(x_c, y_c, x_t, mask, rng, L, mu_data, sigma_data)

        else:

            assert self.input_preprocessing_mode == 'identity'
            mu, sigma = self.model_module()(x_c, y_c, x_t, mask, rng, L)
            sigma = self.post_process_sigma(sigma)
            assert False, 'deprecated'
            # sigma = tree_map(ModelDecorator.sigma_transform_vmap, sigma, mu_data,
            #                      self.sigma_floor.to_dict(), self.sigma_floor_mode.to_dict())

        if self.hermitian_padding != 0:
            assert not isinstance(mu, dict), 'case not yest supported'

            mu = truncate_hermitian_pad(mu, padding=self.hermitian_padding)
            sigma = truncate_hermitian_pad(sigma, padding=self.hermitian_padding)

        return mu, sigma


class CNNDecorator(nn.Module):
    decorated_cnn_module: nn.Module
    input_features: dict

    @nn.compact
    def __call__(self, h):

        if 'pure_delay' in self.input_features.keys() and 'complex_envelope' in self.input_features.keys():
            def stack_along_channel_dimension(pure_delay, complex_envelope):

                sequence_axis = -3
                channel_axis = -1

                assert pure_delay.shape[sequence_axis] == 1
                pure_delay = jnp.concatenate([pure_delay] * complex_envelope.shape[sequence_axis], axis=sequence_axis)

                return jnp.concatenate((pure_delay, complex_envelope), axis=channel_axis)
            h = stack_along_channel_dimension(pure_delay=h['pure_delay'], complex_envelope=h['complex_envelope'])
            return self.decorated_cnn_module()(h)

        elif 'complex_envelope' in self.input_features.keys():
            return self.decorated_cnn_module()(h)

        elif 'pure_delay' in self.input_features.keys():
            return self.decorated_cnn_module()(h)

        else:
            raise ValueError


class DecoderDecorator(nn.Module):
    decorated_decoder_module: nn.Module
    input_features: dict

    @nn.compact
    def __call__(self, f):

        f = self.decorated_decoder_module()(f)

        if 'pure_delay' in self.input_features.keys() and 'complex_envelope' in self.input_features.keys():
            # just what is needed for pure delay, i.e. 4 channels: (mu + sigma) * 2
            pure_delay = f[..., :4]

            # all the rest for complex envelope
            complex_envelope = f[..., 4:]

            # take mean of pure delay values across the frequency bins
            return {'pure_delay': jnp.mean(pure_delay, axis=-3, keepdims=True),
                    'complex_envelope': complex_envelope}

        elif 'complex_envelope' in self.input_features.keys():
            return f

        elif 'pure_delay' in self.input_features.keys():
            return f

        else:
            raise ValueError