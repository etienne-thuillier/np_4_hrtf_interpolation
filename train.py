import logging
import os
from functools import partial
from itertools import cycle
from typing import Callable

import hydra
import jax
import jax.example_libraries.optimizers as jax_opt
import jax.numpy as jnp
import optax
from clu import parameter_overview
from flax.training import checkpoints
from jax import random
from jax.tree_util import tree_map
from ml_collections import FrozenConfigDict
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

import factories.misc
from dataset_statistics import get_statistics
from factories.sconvcnp import INPUT_FEATURES
from utilities.rich_utils import print_config_tree
# from plots import make_qualitative_example_plot_jobs__time_aligned_hrtf
from utilities.utilities import register_resolvers, expand_complex_axis, flatten_dictionary, TrainState

logger = logging.getLogger(__name__)


def init_state(rng, model_factory, dataset_iterable, optimizer_factory, optimizer_hyperparameters):
    init_key, rng = jax.random.split(rng, 2)

    mask, _, x_c, y_c, mu_data, sigma_data = tree_map(lambda z: jnp.array(z), next(dataset_iterable))

    init = model_factory(use_running_stats=True).init(init_key, x_c=x_c, y_c=y_c, x_t=x_c, mask=mask,
                                                      rng=init_key, L=1, mu_data=mu_data,
                                                      sigma_data=sigma_data)
    del x_c, y_c, mask, mu_data, sigma_data

    batch_stats = {'dummy': jnp.zeros((1,),
                                      jnp.float32)}  # dummy placeholder batch_stats array for making script "batch norm agnostic"
    if 'batch_stats' in init.keys():  # case when batch norm is used in model

        batch_stats = init['batch_stats']

    # applying necessary step to be able to change the learning rate between two runs, see
    #     https://github.com/deepmind/optax/discussions/262
    tx = optax.inject_hyperparams(optimizer_factory)(**optimizer_hyperparameters)

    state = TrainState.create(epoch=0,
                              rng=rng,
                              apply_fn=model_factory(use_running_stats=False).apply,
                              eval_fn=model_factory(use_running_stats=True).apply,
                              params=init['params'],
                              tx=tx,
                              batch_stats=batch_stats)

    return state


def train_step(state, mask, w, x_c, y_c, mu_data, sigma_data, rng, L, metrics, reduce_loss, clip_grads):
    """ ... """

    @partial(partial, tree_map)
    def scramble_masked_values(z):
        """ Scramble masked values. This ensures the model is unable to cheat by wrongfully (e.g. because of a bug)
            using masked values. """
        m = jax.lax.broadcast_in_dim(mask,
                                     shape=z.shape,
                                     broadcast_dimensions=list(range(len(mask.shape))))
        return m * z + ~m  # + ~m * jnp.finfo(z.dtype).max

    train_step_sanity_check_input(state=state, x_c=x_c, x_c_weights=w, y_c=y_c, rng=rng, L=L)

    def loss(params, batch_stats):
        # see how to for batch normalization https://github.com/google/flax/issues/932
        (mu, sigma), variables = state.apply_fn({'params': params,
                                                 'batch_stats': batch_stats},
                                                x_c,
                                                scramble_masked_values(y_c),
                                                x_c,
                                                mask,
                                                rng,
                                                L,
                                                mu_data,
                                                sigma_data,
                                                mutable=['batch_stats'])

        train_step_sanity_check_output(mu, sigma)

        ''' as discussed in compute_metrics() we sum here over the dimensions of the input (e.g. pixels of the image)
            in consequence, bigger input dimensions (e.g. larger images) will cause larger learning updates on average
            than small ones, which is what we expect
        '''

        # metrics
        m = {key: metric(w=w, mask=mask, y=y_c, mu=mu, sigma=sigma) for key, metric in metrics.items()}

        # loss
        return reduce_loss(m['per-feature negative log-likelihood']).mean(), \
            (tree_map(lambda z: z.mean(), m), variables['batch_stats'])

    (nll, (m, batch_stats)), grad = jax.value_and_grad(loss, argnums=0, has_aux=True)(state.params,
                                                                                      batch_stats=state.batch_stats)
    if clip_grads:
        grad = jax_opt.clip_grads(grad, 1)

    state = state.replace(batch_stats=batch_stats)

    '''
       ``JAX uses different conventions than TensorFlow for the
         gradients of complex functions. They differ by a conjugate, so we conjugate
         all gradients here in order to make gradient descent work seamlessly. This
         is crucial if there are complex weights in the model, and makes no
         difference for real weights.'' See https://github.com/google/jax/issues/4891
    '''
    grad = tree_map(f=jnp.conj, tree=grad)

    # ''' ... however the parameters are real valued, hence we convert gradient here in order to avoid warning '''
    # values, _ = jax.tree_flatten(grad)
    # assert jnp.all(jnp.array([jnp.min(jnp.abs(z.real/z.imag)) > 1e5 for z in values]))
    # grad = tree_map(lambda z: z.real, grad)

    return m, state.apply_gradients(grads=grad)


def train_step_sanity_check_input(state, x_c, x_c_weights, y_c, rng, L):
    @partial(partial, tree_map)
    def check(y):
        assert len(y.shape) in [6,
                                7], 'expecting array of shape (batch, resolution, resolution, sequence, spin, channel,' \
                                    ' real/imaginary part) or (batch, points, sequence, spin, channel, real/imaginary' \
                                    ' part)'
        assert y.shape[-3] == 1, 'expecting singleton spin dimension'

    check(y_c)


def train_step_sanity_check_output(mu, sigma):
    @partial(partial, tree_map)
    def check(z):
        assert len(z.shape) in [7, 8], 'expecting shape (sample, batch, datapoint, sequence, spin, channel, complex)' \
                                       ' or (sample, batch, resolution, resolution, sequence, spin, channel, complex)'
        assert z.shape[-3] == 1

    check(mu)
    check(sigma)


def train_one_epoch(state, steps_per_epoch, train_iter, reduce_loss, train_L, train_metrics,
                    train_writer, is_profiled, clip_grads):
    for _ in range(steps_per_epoch):

        mask, w, x_c, y_c, mu_data, sigma_data = tree_map(lambda z: jnp.array(z), next(train_iter))

        key, rng_ = random.split(state.rng, 2)

        # assert isinstance(train_metrics, dict) or isinstance(train_metrics, DictConfig)

        metrics, state = jax.jit(train_step, static_argnums=(8, 9, 10, 11))(state=state,
                                                                            mask=mask,
                                                                            w=w,
                                                                            x_c=x_c,
                                                                            y_c=y_c,
                                                                            mu_data=mu_data,
                                                                            sigma_data=sigma_data,
                                                                            rng=key,
                                                                            L=train_L,
                                                                            metrics=FrozenConfigDict(train_metrics),
                                                                            reduce_loss=reduce_loss,
                                                                            clip_grads=clip_grads)
        if is_profiled:
            tree_map(lambda z: z.block_until_ready(), metrics)
            tree_map(lambda z: z.block_until_ready(), state.params)

        # with train_writer.as_default():

        for key, value in flatten_dictionary(metrics).items():
            # tf.summary.scalar(key, value, step=state.step)
            train_writer.add_scalar(tag=key, scalar_value=value.item(), global_step=state.step)

        state = state.replace(rng=rng_)

    return state.replace(epoch=state.epoch + 1)


def train(workdir, rng, steps_per_epoch, model_factory, optimizer_hyperparameters, optimizer_factory,
          train_metrics, train_iterable, reduce_loss, callbacks, train_writer, epochs, profiler_epochs, train_L,
          clip_grads):
    """ ... """

    ''' initialise state '''

    state = init_state(rng=rng,
                       model_factory=model_factory,
                       dataset_iterable=train_iterable,
                       optimizer_factory=optimizer_factory,
                       optimizer_hyperparameters=optimizer_hyperparameters)
    del rng  # for safety to ensure it isn't used downstream since we haven't jax.random.split

    ''' recover checkpoint '''

    state = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(workdir, 'checkpoints'), target=state)
    # overriding checkpoint's optimization hyperparameters by latest specified one
    for key, value in optimizer_hyperparameters.items():
        assert value is not Callable, 'cannot use a schedule as hyper-parameter'
        state.opt_state.hyperparams[key] = value

    parameter_overview.log_parameter_overview(tree_map(expand_complex_axis, state.params))

    print('Dimension of spin-weighted-spherical kernel is (i, o, c, d, f, 2) a.k.a. (spin in, spin out, channel in, '
          'features, # of learnable filter parameters, # of complex plane dimensions)')

    ''' train loop '''

    for _ in range(state.epoch, epochs):

        logger.info('epoch: %i' % state.epoch)

        ''' train step '''

        if state.epoch in profiler_epochs:

            with jax.profiler.trace(log_dir=os.path.join(workdir, 'profiler')):

                state = train_one_epoch(state=state, steps_per_epoch=steps_per_epoch,
                                        train_iter=train_iterable, reduce_loss=reduce_loss,
                                        train_L=train_L, train_metrics=train_metrics,
                                        train_writer=train_writer, is_profiled=True,
                                        clip_grads=clip_grads)

        else:

            state = train_one_epoch(state=state, steps_per_epoch=steps_per_epoch,
                                    train_iter=train_iterable, train_L=train_L, train_metrics=train_metrics,
                                    train_writer=train_writer, is_profiled=False, reduce_loss=reduce_loss,
                                    clip_grads=clip_grads)

        ''' callbacks '''

        for key, value in callbacks.items():
            value(tag=key, state=state)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    print_config_tree(cfg, resolve=True, save_to_file=True)

    jax.config.update("jax_debug_nans", cfg.jax.jax_debug_nans)
    jax.config.update("jax_disable_jit", cfg.jax.jax_disable_jit)
    rng = jax.random.PRNGKey(cfg.seed)

    optimizer = hydra.utils.instantiate(cfg.optimizer)
    metrics = hydra.utils.instantiate(cfg.metrics.metrics)

    if 'input_features' in cfg.model.keys():
        assert cfg.model.input_features == INPUT_FEATURES

    statistics = get_statistics(cfg)

    def make_train_set_loader():
        transform_chain_factory = hydra.utils.instantiate(cfg.data.transforms)
        transforms = transform_chain_factory(statistics=statistics,
                                             observation_count=None,
                                             min_observation_count=0.0,
                                             max_observation_count=cfg.data.max_observation_count,
                                             p_permute_ears=0.5,
                                             p_bernouilli_mask=0.5)

        dataloader_factory = hydra.utils.instantiate(cfg.data.dataloader_factory)
        train_loader = dataloader_factory(split='train',
                                          transforms=transforms,
                                          **cfg.data.splits.train)

        # we defined one epoch as the size of the test set
        steps_per_epoch = cfg.data.splits.test.cardinality // cfg.data.splits.train.batch_size
        return cycle(train_loader), steps_per_epoch

    ''' tensorboard writers '''

    train_writer = SummaryWriter(log_dir=os.path.join(cfg.paths.workdir, 'train'))
    valid_writer = SummaryWriter(log_dir=os.path.join(cfg.paths.workdir, 'valid'))

    ''' callbacks '''

    rng, valid_key, _ = jax.random.split(rng, 3)
    callbacks = factories.misc.make_callbacks(cfg=cfg,
                                              eval_key=valid_key,
                                              metrics=metrics,
                                              writer=valid_writer,
                                              statistics=statistics,
                                              split='valid',
                                              output_path=os.path.join(cfg.paths.workdir, 'valid'))

    train_iterable, steps_per_epoch = make_train_set_loader()

    train(workdir=cfg.paths.workdir,
          rng=rng,
          train_iterable=train_iterable,
          steps_per_epoch=steps_per_epoch,
          model_factory=hydra.utils.instantiate(cfg.model),
          **optimizer(steps_per_epoch=steps_per_epoch),
          train_metrics=hydra.utils.instantiate(cfg.metrics.metrics),
          callbacks=callbacks,
          reduce_loss=lambda z: z,
          train_writer=train_writer,
          epochs=cfg.trainer.epochs,
          profiler_epochs=cfg.trainer.profiler_epochs,
          train_L=1,
          clip_grads=cfg.trainer.clip_grads)


if __name__ == "__main__":
    # torch.multiprocessing.multiprocessing.set_start_method('spawn')

    register_resolvers()

    main()
