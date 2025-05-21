from functools import partial
import flax.linen as nn

from models.modules.misc import ReLU, InSequence, Real, make_sws_transformer, make_2d_equiangular_grid_in_3d_euclidian_space
from models.modules.residual_blocks import SWSPreActivationResidualBlock3D, ResConvBlock
from models.modules.set_convolution import TruncateContextSetDecorator, BatchedSetConvolution, ExponentialKernelSetConvolution, SphericalGaussianKernel, SwapIpsiContraDecorator1, SwapIpsiContraDecorator2
from models.modules.initializers import constant_value_initialiser, softplus_inverse
from models.convcnp import ConvCNP
from models.modules.decorators import ModelDecorator, CNNDecorator, DecoderDecorator

INPUT_FEATURES = dict(complex_envelope=1.0)
REDUCE_LOSS = lambda z: z
NONLINEARITY = ReLU
USE_BATCH_NORM = False
USE_DEPTH_WISE_SEPARABLE_CONVOLUTIONS = False


def get_hermitian_padding(hermitian_padding, n_blocks, filter_taps):
    if hermitian_padding:
        hermitian_padding = n_blocks * (filter_taps - 1) / 2
        assert hermitian_padding == int(hermitian_padding)
        return int(hermitian_padding)
    else:
        return 0


def sconvcnp_factory(use_running_stats, resolution, input_set_convolution_kwargs, max_context_datapoint_count,
					 layers_per_block, features, n_blocks, learnable_swsh_params, filter_taps,
					 latent_set_convolution_kwargs, mlp_layers, sigma_floor, isotropic_complex_noise, rematerialize,
					 swap_ipsi_contra, mlp_block_type, hermitian_padding, input_features, input_preprocessing,
					 noise_mode, **kwargs):
	"""
		factory for spherical convolutional conditional neural process models applicable to off-the-grid data taking
		input features in S^2 (unit sphere embedded in 3d space)

		this version is specialiased for hrtfs

		NB: assumes encoder and decoder has same number of feature channels
	"""

	if isotropic_complex_noise:
		n_output_channels = (2 + 1) * 2  # (2 mu + sigma) * 2 ears
	else:
		n_output_channels = (2 + 2) * 2  # (2 mu + 2 sigma) * 2 ears

	resizer_decoder_layer = latent_set_convolution_kwargs['return_density']

	''' sws transformer '''
	transformer = make_sws_transformer(spins=[(0,)], resolutions=[resolution])

	def cnn_module():

		return partial(CNNDecorator,
					   input_features=input_features,
					   decorated_cnn_module=partial(InSequence.constructor,
													[partial(nn.Conv,
															 features=features,
															 kernel_size=(1, 1, 1, 1),       # shape is (batch, resolution, resolution, sequence, spin, channel)
															 name='resizer')] + \
													[partial(SWSPreActivationResidualBlock3D,
															 features=features,
															 spins=(0,),
															 axis_name=None,
															 transformer=transformer,
															 use_running_stats=use_running_stats,
															 num_filter_params=learnable_swsh_params,
															 num_filter_taps=filter_taps,
															 n_layers=layers_per_block,
															 use_batch_norm=USE_BATCH_NORM,
															 spatial_resample_factor=1,
															 sequence_resample_factor=1,
															 is_depthwise_separable=USE_DEPTH_WISE_SEPARABLE_CONVOLUTIONS)
													 for _ in range(n_blocks)] + \
													[Real],
													rematerialization=rematerialize))

	def pointwise_decoder_module():
		if mlp_block_type == 'residual':
			hidden_blocks = [partial(ResConvBlock,
									 features=features,
									 kernel_size=(1, 1, 1),      # shape is (batch, datapoints, sequence, spin, channel)
									 n_layers=layers_per_block,
									 convolution_kwargs={'use_bias': True},
									 batch_norm_factory=None,    # **** #batch_norm_factory,
									 activation_module=NONLINEARITY,
									 is_depthwise_separable=USE_DEPTH_WISE_SEPARABLE_CONVOLUTIONS,
									 is_dubois=False) for _ in range(mlp_layers)]
		else:
			assert mlp_block_type == 'simple'
			hidden_blocks = sum([[partial(nn.Conv,
										  features=features[-1],
										  kernel_size=(1, 1, 1),
										  padding='SAME',
										  use_bias=True),
								  NONLINEARITY]
								 for _ in range(mlp_layers)], [])

		if resizer_decoder_layer:

			return partial(DecoderDecorator,
						   input_features=input_features,
						   decorated_decoder_module=partial(InSequence.constructor,
															[partial(nn.Conv,
																	 features=features,
																	 kernel_size=(1, 1, 1),  # shape is (batch, datapoints, sequence, spin, channel)
																	 padding='SAME',
																	 use_bias=True,
																	 name='resizer')] + \
															hidden_blocks + \
															[partial(nn.Conv,
																	 features=n_output_channels,
																	 kernel_size=(1, 1, 1),
																	 padding='SAME',
																	 use_bias=True,
																	 name='output')],
															rematerialization=rematerialize))
		else:
			return partial(DecoderDecorator,
						   input_features=input_features,
						   decorated_decoder_module=partial(InSequence.constructor,
															hidden_blocks + \
															[partial(nn.Conv,
																	 features=n_output_channels,
																	 kernel_size=(1, 1, 1),
																	 padding='SAME',
																	 use_bias=True,
																	 name='output')],
															rematerialization=rematerialize))

	def input_set_convolution_module():

		module = partial(TruncateContextSetDecorator,
						 max_context_datapoint_count=max_context_datapoint_count,
						 set_convolution_module=partial(BatchedSetConvolution,
														set_convolution_module=partial(ExponentialKernelSetConvolution,
																					   kernel=SphericalGaussianKernel,
																					   initializer=partial(constant_value_initialiser,
																										   # TODO: make constant a user definable argument
																										   constant=softplus_inverse(2 / resolution / 10)),
																					   return_density=True,
																					   share_scale_across_data_and_density=input_set_convolution_kwargs['share_scale_across_data_and_density']),
														**{key: value for key, value in input_set_convolution_kwargs.items()
														   if key != 'share_scale_across_data_and_density'}))

		if swap_ipsi_contra:
			module = partial(SwapIpsiContraDecorator1, module)

		return module

	def latent_set_convolution_module():
		module = partial(BatchedSetConvolution,
						 set_convolution_module=partial(ExponentialKernelSetConvolution,
														kernel=SphericalGaussianKernel,
														initializer=partial(
															constant_value_initialiser,
															# TODO: make constant a user definable argument
															constant=softplus_inverse(2 / resolution / 10)),
														return_density=
														latent_set_convolution_kwargs[
															'return_density'],
														share_scale_across_data_and_density=
														latent_set_convolution_kwargs[
															'share_scale_across_data_and_density']),
						 **{key: value for key, value in
							latent_set_convolution_kwargs.items()
							if key not in ['return_density',
										   'share_scale_across_data_and_density']})

		if swap_ipsi_contra:
			module = partial(SwapIpsiContraDecorator2, module)

		return module

	return ModelDecorator(input_preprocessing_mode=input_preprocessing,
						  noise_mode=noise_mode,
						  hermitian_padding=get_hermitian_padding(hermitian_padding=hermitian_padding,
																  n_blocks=n_blocks,
																  filter_taps=filter_taps),
						  model_module=partial(ConvCNP,
											   grid=make_2d_equiangular_grid_in_3d_euclidian_space(resolution=resolution),
											   input_set_convolution_module=input_set_convolution_module(),
											   cnn_module=cnn_module(),
											   latent_set_convolution_module=latent_set_convolution_module(),
											   pointwise_decoder_module=pointwise_decoder_module(),
											   mu_transform=lambda mu: mu,
											   sigma_transform=lambda sigma: sigma_floor + (1 - sigma_floor) * nn.softplus(sigma),
											   isotropic_complex_noise=isotropic_complex_noise,
											   calibration_set_convolution_module=None))
