from functools import partial
import os.path
import hydra
import logging
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

import jax
from jax.tree_util import tree_map
from flax.training import checkpoints
from flax.core.frozen_dict import FrozenDict

from dataset_statistics import get_statistics
import factories.misc
from utilities.utilities import register_resolvers, TrainState
from utilities.rich_utils import print_config_tree


logger = logging.getLogger(__name__)


def predict(apply_fn, params, batch_stats, x_c, y_c, x_t, mask, rng, L, mu_data, sigma_data):
    @partial(partial, tree_map)
    def scramble_masked_values(z):
        """ make sure the model is not cheating by wrongfully using masked values """
        m = jax.lax.broadcast_in_dim(mask,
                                     shape=z.shape,
                                     broadcast_dimensions=list(range(len(mask.shape))))
        return m * z + ~m

    predict_sanity_check_input(x_c, y_c, x_t, mask, rng, L)

    (mu, sigma), state = apply_fn({'params': params,
                                   'batch_stats': batch_stats},
                                  x_c,
                                  scramble_masked_values(y_c),
                                  x_t, mask, rng, L,
                                  mu_data,
                                  sigma_data,
                                  mutable=['batch_stats', 'latents'])

    predict_sanity_check_output(mu, sigma)

    if 'latents' in state.keys():
        latents = state['latents']
    else:
        latents = None

    return mu, sigma, latents


def predict_sanity_check_input(x_c, y_c, x_t, mask, rng, L):
    @partial(partial, tree_map)
    def check(z):
        assert len(z.shape) in [6,
                                7], 'expecting array of shape (batch, resolution, resolution, sequence, spin, channel,' \
                                    ' real/imaginary part) or (batch, points, sequence, spin, channel, real/imaginary part)'
        assert z.shape[-3] == 1, 'expecting singleton spin dimension'

    check(y_c)
    assert len(mask.shape) == len(x_c.shape[:-1]), 'expecting array of shape (batch, datapoints)'


def predict_sanity_check_output(mu, sigma):
    @partial(partial, tree_map)
    def check(z):
        assert len(z.shape) in [7, 8], 'expecting array of shape (sample, batch, resolution, resolution, sequence, ' \
                                       'spin, channel, ***) or (sample, batch, points, sequence, spin, channel, ***)'
        assert z.shape[-3] == 1, 'expecting singleton spin dimension'

    check(mu)
    check(sigma)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg : DictConfig) -> None:

    assert os.path.exists(cfg.paths.workdir), "experiment folder {cfg.paths.workdir} does not exist"

    print_config_tree(cfg, resolve=True, save_to_file=True)

    jax.config.update("jax_debug_nans", cfg.jax.jax_debug_nans)
    jax.config.update("jax_disable_jit", cfg.jax.jax_disable_jit)
    rng = jax.random.PRNGKey(cfg.seed)

    metrics = hydra.utils.instantiate(cfg.metrics.metrics)
    statistics = get_statistics(cfg)

    ''' tensorboard writers '''

    test_writer = SummaryWriter(log_dir=os.path.join(cfg.paths.workdir, 'test'))

    ''' callbacks '''

    logger.info('removed all callbacks but "metric vs sample count"')

    cfg.callbacks = DictConfig({'metric vs sample count': cfg.callbacks['metric vs sample count']})
    cfg.callbacks['metric vs sample count']['period'] = 1

    rng, _, test_key = jax.random.split(rng, 3)
    callbacks = factories.misc.make_callbacks(cfg=cfg,
                                              eval_key=test_key,
                                              metrics=metrics,
                                              writer=test_writer,
                                              statistics=statistics,
                                              split='test')

    logger.info('recovering checkpoint')

    if cfg.checkpoint is None:
        state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(cfg.paths.workdir, 'checkpoints'), target=None)
    else:
        state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(cfg.paths.workdir,
                                                                     'checkpoints',
                                                                     cfg.checkpoint), target=None)

    if state is None:
        logger.warning('no checkpoints found, using dummy state')
        batch_stats, epoch, opt_state, params, rng, step = FrozenDict(dict()), 0, dict(), FrozenDict(dict()), rng, 0
    else:
        logger.info('found checkpoint')
        batch_stats, epoch, opt_state, params, rng, step = state['batch_stats'], state['epoch'], state['opt_state'], \
            state['params'], state['rng'], state['step']
        params, batch_stats = FrozenDict(state['params']), FrozenDict(state['batch_stats'])

    model_factory = hydra.utils.instantiate(cfg.model)
    apply_fn = model_factory(use_running_stats=False).apply
    eval_fn = model_factory(use_running_stats=True).apply
    state = TrainState(batch_stats=batch_stats, epoch=epoch, params=params, apply_fn=apply_fn, eval_fn=eval_fn,
                       tx=None, opt_state=opt_state, rng=rng, step=step)

    for key, value in callbacks.items():
        logger.info("running callback {}".format(key))
        value(tag=key, state=state)


if __name__ == "__main__":

    # torch.multiprocessing.multiprocessing.set_start_method('spawn')

    register_resolvers()

    main()