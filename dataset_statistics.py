import logging
import pickle
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

# from plots import make_qualitative_example_plot_jobs__time_aligned_hrtf
from utilities.utilities import expand_complex_axis, collapse_complex_axis

logger = logging.getLogger(__name__)


def make_dataset_statistics(train_set_loader, destination_path=None):
    # TODO: we ended up hard-coding these options for simplicity
    data_augmentations = ['permute_ears']
    isotropic_complex_noise = False

    x, y = next(iter(train_set_loader))

    def percentile(z):
        magnitude = np.sqrt((z ** 2).sum(-1))
        return np.percentile(magnitude.flatten(), q=99)

    peak_magnitude = tree_map(percentile, y)

    if 'permute_ears' in data_augmentations:
        y = tree_map(partial(constellation_flip_right_ear, x=x), y)
        y = tree_map(lambda z: jnp.concatenate((z[..., :1, :], z[..., 1:, :]), axis=0), y)

    mu_data = tree_map(lambda z: np.mean(z, axis=0), y)
    if isotropic_complex_noise:
        sigma_data = tree_map(lambda z: expand_complex_axis(np.std(collapse_complex_axis(z), axis=0)), y)
    else:
        sigma_data = tree_map(lambda z: np.std(z, axis=0), y)
        # 0 values occur in imaginary part, e.g. @ DC
        sigma_data = tree_map(lambda z: z + jnp.finfo(z.dtype).eps, sigma_data)

    del y, train_set_loader
    if 'permute_ears' in data_augmentations:
        # creating right ear statistics from left ear

        # flipping back the right ear around median plane
        mu_data = tree_map(lambda z: jnp.concatenate((z, z), axis=-2), mu_data)
        sigma_data = tree_map(lambda z: jnp.concatenate((z, z), axis=-2), sigma_data)

        # re-introducting batch dimension
        mu_data = tree_map(lambda z: jnp.expand_dims(z, axis=0), mu_data)
        sigma_data = tree_map(lambda z: jnp.expand_dims(z, axis=0), sigma_data)

        # flipping back the right ear around median plane
        mu_data = tree_map(partial(constellation_flip_right_ear, x=x[:1]), mu_data)
        sigma_data = tree_map(partial(constellation_flip_right_ear, x=x[:1]), sigma_data)

        # removing batch dimension
        mu_data = tree_map(lambda z: z[0], mu_data)
        sigma_data = tree_map(lambda z: z[0], sigma_data)

    statistics = dict(peak_magnitude=peak_magnitude.item(),
                      mu_data=np.array(mu_data),
                      sigma_data=np.array(sigma_data))

    if destination_path is not None:
        with open(destination_path, 'wb') as f:
            pickle.dump(obj=statistics, file=f)

    return statistics


def constellation_flip_right_ear(y, x):
    assert y.shape[-2] == 2, 'expecting two left/right ear channels'
    assert len(x.shape[1:-1]) == 1, 'expecting (1d) list of coordinates'
    assert len(x.shape[:-1]) + len(['sequence', 'spin', 'channel', 'complex']) == len(y.shape), \
        'expecting y with shape (batch, *mask.shape, sequence, spin, channel, complex)'
    assert y.shape[:-len(['sequence', 'spin', 'channel', 'complex'])] == x.shape[:-1], \
        'expecting y with shape (batch, *mask.shape, sequence, spin, channel, complex)'

    def get_permutation_indices(x_):
        indices = list()
        for x__ in x_:
            i = jnp.where(jax.vmap(jnp.allclose, in_axes=(None, 0))(jnp.stack((x__[0], -x__[1], x__[2])), x_))
            assert len(i) == 1
            indices += i
        return jnp.concatenate(indices)

    assert jnp.all(x[0] == x)

    i = get_permutation_indices(x[0])
    assert jnp.all(i[i] == jnp.arange(
        i.shape[0])), 'expecting permutation indices to carry-out two-by-two swaps, something is fishy here...'
    return jnp.concatenate((y[:, :, ..., :1, :],
                            y[:, i, ..., 1:, :]), axis=-2)


def get_statistics(cfg):
    """ get statistics from dataloader """
    dataloader_factory = hydra.utils.instantiate(cfg.data.dataloader_factory)
    transforms_4_statistics = hydra.utils.instantiate(cfg.data.transforms_4_statistics)

    dataloader_4_statistics = dataloader_factory(split='train',
                                                 cardinality=None,  # whole dataset, no repeats
                                                 seed=cfg.seed,
                                                 batch_size=None,  # single batch containing whole dataset
                                                 shuffle=False,
                                                 drop_last=False,
                                                 transforms=transforms_4_statistics)

    return make_dataset_statistics(train_set_loader=dataloader_4_statistics)
