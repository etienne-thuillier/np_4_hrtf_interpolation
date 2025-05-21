from functools import partial, reduce
from typing import Sequence
import operator
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

import spin_spherical_cnns.spin_spherical_harmonics as ssh
import spin_spherical_cnns.sphere_utils as sphere_utils

from utilities.coordinate_system import sph2cart__sfs


class InSequence(nn.Module):
    modules: Sequence[nn.Module]

    @nn.nowrap
    @staticmethod
    def constructor(modules, rematerialization, **kwargs):

        def recursive_checkpoint(modules):
            """ inspired from example 5. in https://github.com/google/jax/pull/1749 """

            if len(modules) == 1:

                return modules[0]

            elif len(modules) == 2:

                block_1, block_2 = modules

                return partial(InSequence, modules=[nn.remat(block_1), block_2])

            else:

                block_1 = recursive_checkpoint(modules[:len(modules) // 2])
                block_2 = recursive_checkpoint(modules[len(modules) // 2:])

                return partial(InSequence, modules=[nn.remat(block_1), block_2])

        if rematerialization:

            modules = recursive_checkpoint(modules)

            return InSequence([modules], **kwargs)

        else:

            return InSequence(modules, **kwargs)

    @nn.compact
    def __call__(self, *args):

        def call(module, args_):

            if not isinstance(args_, tuple):
                args_ = (args_,)

            return module()(*args_)

        for module in self.modules:
            args = call(module, args)

        return args


class Real(nn.Module):

    @nn.compact
    def __call__(self, x):
        return x.real


class ReLU(nn.Module):
	""" wrapper of relu function into a module, as required when rematerialization is carried out in InSequence
	    modules
	"""
	@nn.compact
	def __call__(self, x):
		return nn.relu(x)


def make_2d_equiangular_grid_in_3d_euclidian_space(resolution):

    longitude, colatitude = sphere_utils.make_equiangular_grid(resolution=resolution)
    grid = jnp.stack(sph2cart__sfs(alpha=longitude, beta=colatitude, r=1.0), axis=-1)

    return grid


def make_sws_transformer(spins, resolutions):
    ''' flatten spins '''
    all_spins = reduce(operator.concat, spins)

    ''' make transformer: this will carry out pre-computations that will be cached to 
        speed up sw-spherical transforms '''
    transformer = ssh.SpinSphericalFourierTransformer(resolutions=np.unique(resolutions),
                                                      spins=np.unique(all_spins))

    return transformer