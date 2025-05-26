from functools import partial

from models.modules.decorators import ModelDecorator
from models.spherical_barycentric_interpolator import BarycentricInterpolator


def barycentric_factory(input_preprocessing, noise_mode, **kwargs):
    """ ... """

    return ModelDecorator(input_preprocessing_mode=input_preprocessing,
                          noise_mode=noise_mode,
                          hermitian_padding=0,
                          model_module=partial(BarycentricInterpolator, sigma_value=1e-3))
