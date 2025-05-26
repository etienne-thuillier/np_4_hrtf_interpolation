from typing import Dict, Callable, Optional, Sequence, Any

import flax.linen as nn

from models.modules.spherical_convolution import (SpinSphericalPlanarConvolution,
                                                  SpinSphericalPlanarBatchNormMagnitudeNonlin,
                                                  SpinSphericalPlanarMagnitudeNonlin)
from spin_spherical_cnns.spin_spherical_harmonics import SpinSphericalFourierTransformer


class ResConvBlock(nn.Module):
    """
    Copied and adapted from neural_process_family/npf/architectures/cnn.py: ResConvBlock

    Convolutional block inspired by the pre-activation Resnet [1] and depthwise separable
    convolutions [2].

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """
    features: int
    kernel_size: tuple
    n_layers: int
    convolution_kwargs: Dict
    activation_module: Callable
    batch_norm_factory: Optional[Callable] = None
    is_depthwise_separable: Optional[bool] = True
    is_dubois: Optional[bool] = False

    def convolve_dubois(self, x, y):

        if self.is_depthwise_separable:

            n_channels = y.shape[-1]

            # depth-wise convolution

            y = nn.Conv(features=n_channels,
                        kernel_size=self.kernel_size,
                        feature_group_count=n_channels,
                        **self.convolution_kwargs)(y)

            y = x + y

            # point-wise convolution

            kwargs = {key: value for key, value in self.convolution_kwargs.items()
                      if key not in ['features', 'kernel_size', 'strides', 'padding', 'input_dilation',
                                     'kernel_dilation', 'feature_group_count']}

            return nn.Conv(features=self.features,
                           kernel_size=tuple([1] * len(self.kernel_size)),
                           strides=1,
                           padding='SAME',
                           input_dilation=1,
                           kernel_dilation=1,
                           feature_group_count=1,
                           **kwargs)(y)
        else:

            assert False, 'not coded'

    def convolve(self, y):

        if self.is_depthwise_separable:

            n_channels = y.shape[-1]

            # depth-wise convolution

            y = nn.Conv(features=n_channels,
                        kernel_size=self.kernel_size,
                        feature_group_count=n_channels,
                        padding='SAME',
                        **self.convolution_kwargs)(y)

            # point-wise convolution

            kwargs = {key: value for key, value in self.convolution_kwargs.items()
                      if key not in ['features', 'kernel_size', 'strides', 'padding', 'input_dilation',
                                     'kernel_dilation', 'feature_group_count']}

            return nn.Conv(features=self.features,
                           kernel_size=tuple([1] * len(self.kernel_size)),
                           strides=1,
                           padding='SAME',
                           input_dilation=1,
                           kernel_dilation=1,
                           feature_group_count=1,
                           **kwargs)(y)
        else:

            return nn.Conv(features=self.features,
                           kernel_size=self.kernel_size,
                           **self.convolution_kwargs)(y)

    def batch_norm(self, y):

        if self.batch_norm_factory is not None:
            y = self.batch_norm_factory()(y)

        return y

    def two_layers(self, x):

        y = self.batch_norm(x)
        y = self.activation_module()(y)
        y = self.convolve(y)

        y = self.batch_norm(y)
        y = self.activation_module()(y)

        if self.is_dubois:

            return self.convolve_dubois(x, y)

        else:

            y = self.convolve(y)
            return x + y

    def single_layer(self, x):

        y = self.batch_norm(x)
        y = self.activation_module()(y)

        if self.is_dubois:

            return self.convolve_dubois(x, y)

        else:

            y = self.convolve(y)
            return x + y

    @nn.compact
    def __call__(self, x):

        assert self.n_layers in [1, 2]

        if self.n_layers == 2:

            return self.two_layers(x)

        else:

            assert self.n_layers == 1
            return self.single_layer(x)


class SWSPreActivationResidualBlock3D(nn.Module):
    """
        Convolutional block inspired by the pre-activation Resnet [1] and depthwise separable
        convolutions [2].

        Same as SWSPreActivationResidualBlock but for three-dimensional signals of form

            S^2 \times \mathds{R} -> \mathds{K},            K \in {\mathds{R}, \mathds{C}},

        where \mathds{R} component of domain corresponds to time for example.

        ----------
        [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
            in deep residual networks. In European conference on computer vision
            (pp. 630-645). Springer, Cham.

        [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
            convolutions. In Proceedings of the IEEE conference on computer vision
            and pattern recognition (pp. 1251-1258).
    """

    features: int
    spins: Sequence[int]
    axis_name: Any
    transformer: SpinSphericalFourierTransformer
    use_running_stats: bool
    num_filter_params: Optional[int] = None
    num_filter_taps: Optional[int] = 3

    n_layers: Optional[int] = 2
    use_batch_norm: Optional[bool] = True

    spatial_resample_factor: Optional[float] = 1.0
    sequence_resample_factor: Optional[float] = 1.0

    is_depthwise_separable: Optional[bool] = 1.0

    def convolve(self, x, spatial_resample_factor, sequence_resample_factor):

        if self.is_depthwise_separable:

            assert False, 'PointwiseConv4SWSF isn''t yet implemented'

            n_channels = x.shape[-1]

            y = SpinSphericalConvolution3D(features=n_channels,
                                           spins_in=self.spins,
                                           spins_out=self.spins,
                                           transformer=self.transformer,
                                           num_filter_params=self.num_filter_params,
                                           num_filter_taps=self.num_filter_taps,
                                           spatial_resample_factor=spatial_resample_factor,
                                           sequence_resample_factor=sequence_resample_factor,
                                           is_depthwise=True)(x)

            return PointwiseConv4SWSF(features=self.features)(y)

        else:

            return SpinSphericalPlanarConvolution(features=self.features,
                                                  spins_in=self.spins,
                                                  spins_out=self.spins,
                                                  transformer=self.transformer,
                                                  num_filter_params=self.num_filter_params,
                                                  num_filter_taps=self.num_filter_taps,
                                                  spatial_resample_factor=spatial_resample_factor,
                                                  sequence_resample_factor=sequence_resample_factor,
                                                  is_depthwise=False)(x)

    def nonlinearity(self, x):

        if self.use_batch_norm:

            return SpinSphericalPlanarBatchNormMagnitudeNonlin(spins=self.spins,
                                                               axis_name=self.axis_name,
                                                               use_running_stats=self.use_running_stats)(x)

        else:

            return SpinSphericalPlanarMagnitudeNonlin(spins=self.spins)(x)

    def two_layers(self, x):

        y = self.nonlinearity(x)

        y = self.convolve(y,
                          spatial_resample_factor=self.spatial_resample_factor,
                          sequence_resample_factor=self.sequence_resample_factor)

        y = self.nonlinearity(y)

        y = self.convolve(y,
                          spatial_resample_factor=1,
                          sequence_resample_factor=1)

        if self.spatial_resample_factor != 1 or self.sequence_resample_factor != 1:
            x = self.convolve(x,
                              spatial_resample_factor=self.spatial_resample_factor,
                              sequence_resample_factor=self.sequence_resample_factor)

        return x + y

    def single_layer(self, x):

        y = self.nonlinearity(x)

        y = self.convolve(y,
                          spatial_resample_factor=self.spatial_resample_factor,
                          sequence_resample_factor=self.sequence_resample_factor)

        if self.spatial_resample_factor != 1 or self.sequence_resample_factor != 1:
            x = self.convolve(x,
                              spatial_resample_factor=self.spatial_resample_factor,
                              sequence_resample_factor=self.sequence_resample_factor)

        return x + y

    @nn.compact
    def __call__(self, x):

        assert self.n_layers in [1, 2]

        if self.n_layers == 2:

            return self.two_layers(x)

        else:

            assert self.n_layers == 1
            return self.single_layer(x)
