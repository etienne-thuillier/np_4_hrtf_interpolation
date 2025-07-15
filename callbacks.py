import heapq
import logging
import os
import pickle
from functools import partial
from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from jax.tree_util import tree_map

from metrics import mv_mse
from test import predict
from utilities.plots import make_metric_vs_sample_count_graph
from utilities.utilities import remove_dir_recursively, flatten_dictionary

CheckpointEntry = Tuple[float, str]  # (score, path)
logger = logging.getLogger(__name__)


def filter_callback(state, callback, period, tag=None):
    """
    :param state:           a dictionary-like structure with field "epoch"
    :param callback:        function with single argument state
    :param period:          a positive-valued integer
    :param tag:           a string describing the callback (can be None)
    :return:
    """

    if state.epoch % period == 0:
        if tag is not None:
            logger.info(tag)
        callback(state)
    else:
        pass


def evaluate(rng, eval_iter, state, L, metrics):
    """ predicts and evaluate metrics, returning a scalar per batch element """

    def f(key, mask, x, y, mu_data, sigma_data):

        return predict(apply_fn=state.eval_fn,
                       params=state.params,
                       batch_stats=state.batch_stats,
                       x_c=x, y_c=y, x_t=x,
                       mask=mask, rng=key, L=L,
                       mu_data=mu_data, sigma_data=sigma_data)

    m = None
    for batch in eval_iter:

        mask, w, x, y, mu_data, sigma_data = tree_map(lambda z: jnp.array(z), batch)

        sample_rng, rng = jax.random.split(rng, 2)

        mu, sigma, _ = jax.jit(f)(key=sample_rng, mask=mask, x=x, y=y, mu_data=mu_data, sigma_data=sigma_data)

        m_ = {key: metric(w=w, mask=mask, y=y, mu=mu, sigma=sigma) for key, metric in metrics.items()}

        if m is None:
            m = m_

        else:
            m = tree_map(lambda z1, z2: jnp.concatenate((z1, z2), axis=0), m, m_)

    return m


def log_evaluation(state, rng, writer, eval_iter, L, metrics, checkpoint_manager):
    m = evaluate(rng=rng, eval_iter=eval_iter, state=state, L=L, metrics=metrics)

    metric_means = {key: value.mean().item() for key, value in flatten_dictionary(m).items()}

    for key, value in metric_means.items():
        writer.add_scalar(tag=key, scalar_value=value, global_step=state.step)
        print('eval: epoch {} {} {:.4f}'.format(state.epoch, key, value))

    if checkpoint_manager is not None:
        checkpoint_manager.maybe_save_checkpoint(state, metrics=metric_means)

    if 'psi_in' in state.params.keys():

        if 'softplus^{-1}(scale)' in state.params['psi_in'].keys():
            writer.add_histogram(tag='psi_in/scale',
                                 values=np.array(nn.softplus(state.params['psi_in']['softplus^{-1}(scale)'])),
                                 global_step=state.step)

    if 'psi_out' in state.params.keys():

        if 'softplus^{-1}(scale)' in state.params['psi_out'].keys():
            writer.add_histogram(tag='psi_out/scale',
                                 values=np.array(nn.softplus(state.params['psi_out']['softplus^{-1}(scale)'])),
                                 global_step=state.step)


def evaluate_uncertainty_miscalibration__helper(state, key, writer, eval_iter, n_bins,
                                                feature_decimating_factors, eval_batch):
    rng = key

    assert np.log2(n_bins) == int(np.log2(n_bins))
    assert np.log2(eval_batch) == int(np.log2(eval_batch))

    # n_batches = n_datapoints // config.valid.batch

    def rmv_rmse_local(y, mu, sigma):

        def rmv_rmse_wrapper(y_, mu_, sigma_):

            mv, mse = mv_mse(y_, mu_, sigma_, n_bins)
            rmv, rmse = jnp.sqrt(mv), jnp.sqrt(mse)

            return (rmv, rmse)

        vmap_trailing_dimension = lambda f: jax.vmap(f, in_axes=(-1, -1, -1), out_axes=(-1, -1))

        rmv_rmse_spatial_0 = vmap_trailing_dimension(rmv_rmse_wrapper)
        global x
        if len(x.shape) == 4:
            rmv_rmse_spatial_1 = vmap_trailing_dimension(rmv_rmse_spatial_0)
        else:
            assert len(x.shape) == 3, 'expecting shape (batch, # points, # of 3D coordinates)'
            rmv_rmse_spatial_1 = rmv_rmse_spatial_0

        rmv_rmse_sequence = vmap_trailing_dimension(rmv_rmse_spatial_1)
        rmv_rmse_spin = vmap_trailing_dimension(rmv_rmse_sequence)
        rmv_rmse_channel = vmap_trailing_dimension(rmv_rmse_spin)

        if sigma.shape[-1] == mu.shape[-1]:
            rmv_rmse_complex = vmap_trailing_dimension(rmv_rmse_channel)
        else:
            assert sigma.shape[-1] == 1 and mu.shape[-1] == 2, 'expecting real-valued sigma and complex-valued mu'
            rmv_rmse_complex = jax.vmap(rmv_rmse_channel, in_axes=(-1, -1, None), out_axes=(-1, -1))
            sigma = sigma[..., 0]

        return rmv_rmse_complex(y, mu, sigma)

    def decimate_feature_dimensions(rng_, z):

        assert len(feature_decimating_factors) == 4
        space_decimation, sequence_decimation, spin_decimation, channel_decimation = feature_decimating_factors

        @partial(partial, tree_map)
        def f(z__):

            global x
            if len(x.shape) == 3:

                assert x.shape[:-1] == z__.shape[:-4]
                _, space_, sequence_, spin_, channel_, complex_ = z__.shape

                space_rng, sequence_rng, spin_rng, channel_rng = jax.random.split(rng_, 4)

                i_space = jax.random.choice(space_rng, space_, (space_ // min(space_, space_decimation),))
                i_sequence = jax.random.choice(sequence_rng, sequence_,
                                               (sequence_ // min(sequence_, sequence_decimation),))
                i_spin = jax.random.choice(spin_rng, spin_, (spin_ // min(spin_decimation, spin_),))
                i_channel = jax.random.choice(channel_rng, channel_, (channel_ // min(channel_, channel_decimation),))

                def decimate(z_):

                    return z_[:, i_space, ...][:, :, i_sequence, ...][:, :, :, i_spin, ...][:, :, :, :, i_channel, :]

            else:

                assert len(x.shape) == 4
                assert x.shape[:-1] == z__.shape[:-4]

                _, space_0_, space_1_, sequence_, spin_, channel_, complex_ = z__.shape

                space_0_rng, space_1_rng, sequence_rng, spin_rng, channel_rng = jax.random.split(rng_, 5)

                space_decimation_0 = int(np.round(np.sqrt(space_decimation)))
                space_decimation_1 = max(1, int(space_decimation / space_decimation_0))

                i_space_0 = jax.random.choice(space_0_rng, space_0_, (space_0_ // min(space_0_, space_decimation_0),))
                i_space_1 = jax.random.choice(space_1_rng, space_1_, (space_1_ // min(space_1_, space_decimation_1),))
                i_sequence = jax.random.choice(sequence_rng, sequence_,
                                               (sequence_ // min(sequence_, sequence_decimation),))
                i_spin = jax.random.choice(spin_rng, spin_, (spin_ // min(spin_, spin_decimation),))
                i_channel = jax.random.choice(channel_rng, channel_, (channel_ // min(channel_, channel_decimation),))

                def decimate(z_):

                    return z_[:, i_space_0, ...][:, :, i_space_1, ...][:, :, :, i_sequence, ...][:, :, :, :, i_spin,
                           ...][:, :, :, :, :, i_channel, :]

            return decimate(z__)

        return f(z)

    cardinality = 0
    for mask, w, x_, y_, mu_data_, sigma_data_ in iter(eval_iter):
        assert len(mask.shape[1:]) == 1, 'grid shape not yet supported'
        cardinality += mask.shape[0]

    y = None
    mu = None
    sigma = None
    global x

    def f(key, mask, x, y, mu_data, sigma_data):

        mu, sigma, _ = predict(apply_fn=state.eval_fn,
                               params=state.params,
                               batch_stats=state.batch_stats,
                               x_c=x, y_c=y, x_t=x,
                               mask=mask, rng=key,
                               L=1,  # ****
                               mu_data=mu_data, sigma_data=sigma_data)

        return mu, sigma

    for batch in eval_iter:

        mask, w, x, y_, mu_data_, sigma_data_ = tree_map(lambda z: jnp.array(z), batch)

        sample_rng, rng = jax.random.split(rng, 2)

        mu_, sigma_ = jax.jit(f)(key=sample_rng, mask=mask, x=x, y=y_,
                                 mu_data=mu_data_, sigma_data=sigma_data_)

        # TODO: map to error space here?
        # y_ = FLAGS.config.to_error_space(y_)
        # mu_, sigma_ = FLAGS.config.to_error_space(mu_, sigma_)

        # here we use state's rng instead of that provided from seed since we welcome that different features be
        # selected across iterations

        if feature_decimating_factors != (1, 1, 1, 1):
            y_ = decimate_feature_dimensions(rng, y_)
            mu_ = jax.vmap(decimate_feature_dimensions, in_axes=(None, 0))(rng, mu_)
            sigma_ = jax.vmap(decimate_feature_dimensions, in_axes=(None, 0))(rng, sigma_)

        y_ = jax.device_put(y_, jax.devices("cpu")[0])
        mu_ = jax.device_put(mu_, jax.devices("cpu")[0])
        sigma_ = jax.device_put(sigma_, jax.devices("cpu")[0])

        if mu is None:

            assert sigma is None
            y = y_
            mu = mu_
            sigma = sigma_

        else:

            y = tree_map(lambda z1, z2: jnp.concatenate((z1, z2), axis=0), y, y_)
            mu = tree_map(lambda z1, z2: jnp.concatenate((z1, z2), axis=1), mu, mu_)
            sigma = tree_map(lambda z1, z2: jnp.concatenate((z1, z2), axis=1), sigma, sigma_)

    @partial(partial, tree_map)
    def collapse_sample_dimension(z):
        assert z.shape[0] == 1
        return z[0]

    @partial(partial, tree_map)
    def reshape(z):
        return jnp.reshape(z, (-1, 1, *z.shape[2:]))

    y, mu, sigma = reshape(y), reshape(collapse_sample_dimension(mu)), reshape(collapse_sample_dimension(sigma))
    rmv_rmse_ = tree_map(rmv_rmse_local, y, mu, sigma)

    if writer is not None:

        if isinstance(rmv_rmse_, dict):

            for key, item in rmv_rmse_.items():
                writer.add_scalar(tag='mean_ence/' + key,
                                  scalar_value=np.array((np.abs(item[1] - item[0]) / item[0]).mean()),
                                  global_step=state.step)
                # description='mean ence value across output features')

                writer.add_histogram(tag='ence/' + key,
                                     values=np.array((np.abs(item[1] - item[0]) / item[0]).mean(axis=0).flatten()),
                                     global_step=state.step)
                # description='ence histogram of output features')

                cd = np.abs(20 * np.log10(item[1] / item[0]))
                writer.add_scalar(tag='mcd/' + key,
                                  scalar_value=np.array(cd.mean()),
                                  global_step=state.step,
                                  description='mean calibration distance: mean( |10 log10 (mse/mpv)|  ), where mse is mean square error and mpv is mean predicted variance')
                writer.add_histogram(tag='cd',
                                     values=np.array(cd.flatten()),
                                     global_step=state.step)
            # description='calibration distance: |10 log10 (mse/mpv)|')

        else:

            ence = (np.abs(rmv_rmse_[1] - rmv_rmse_[0]) / rmv_rmse_[0]).mean()
            writer.add_scalar(tag='mean_ence',
                              scalar_value=np.array(ence),
                              global_step=state.step)
            # description='mean ence value across output features')
            writer.add_histogram(tag='ence',
                                 values=np.array(
                                     (np.abs(rmv_rmse_[1] - rmv_rmse_[0]) / rmv_rmse_[0]).mean(axis=0).flatten()),
                                 global_step=state.step)
            # description='ence histogram of output features')

            cd = np.abs(20 * np.log10(rmv_rmse_[1] / rmv_rmse_[0]))
            writer.add_scalar(tag='mcd',
                              scalar_value=np.array(cd.mean()),
                              global_step=state.step)
            # description='mean calibration distance: mean( |10 log10 (mse/mpv)|  ), where mse is mean square error and mpv is mean predicted variance')
            writer.add_histogram(tag='cd',
                                 values=np.array(cd.flatten()),
                                 global_step=state.step)
        # description='calibration distance: |10 log10 (mse/mpv)|')

    return rmv_rmse_


def evaluate_at_sample_counts(rng, state, L, metrics, count_2_dataset):
    """ note, this only works with mask_generator == draw_uniform_s2_grid_wrapper """

    f = partial(evaluate, rng=rng, state=state, L=L, metrics=metrics)

    m = None
    for sample_count, dataset in count_2_dataset.items():

        logger.info('sample count: %s' % sample_count)

        m_ = f(eval_iter=dataset)

        m_ = tree_map(lambda z: jnp.expand_dims(z, axis=0), m_)
        if m is None:
            m = m_
        else:
            m = tree_map(lambda z1, z2: jnp.concatenate((z1, z2), axis=0), m, m_)

    return [int(sample_count) for sample_count in count_2_dataset.keys()], m


def plot_metrics_vs_sample_count(state, key, L, metrics, count_2_dataset_iterables, images_writer, output_path):
    sample_counts, metrics = evaluate_at_sample_counts(rng=key,
                                                       state=state,
                                                       L=L,
                                                       metrics=metrics,
                                                       count_2_dataset=count_2_dataset_iterables)

    # save result for offline plotting the case being
    if output_path is not None:
        with open(os.path.join(output_path, f"metrics_vs_sample_count_{state.step}.pickle"), 'wb') as f:
            pickle.dump(dict(sample_counts=sample_counts, metrics=metrics), f, protocol=pickle.HIGHEST_PROTOCOL)

    images = make_metric_vs_sample_count_graph(sample_counts=sample_counts, metrics=metrics)

    for key, image in images.items():
        images_writer.add_image(tag=key.replace(' ', '_'), img_tensor=image.transpose(2, 0, 1), global_step=state.step)


class TopKCheckpointManager:

    def __init__(self, ckpt_dir: str, k: int = 5, metric: str = "loss", maximize: bool = False):
        self.ckpt_dir = ckpt_dir
        self.k = k
        self.maximize = maximize
        self.metric_str = metric
        self.checkpoints: List[CheckpointEntry] = []

        os.makedirs(ckpt_dir, exist_ok=True)

    def maybe_save_checkpoint(self, state: Any, metrics: Dict):

        score = metrics[self.metric] if self.maximize else -metrics[self.metric_str]

        def save_checkpoint():

            checkpoints.save_checkpoint(ckpt_dir=self.ckpt_dir,
                                        target=state,
                                        step=state.step,
                                        prefix=f"checkpoint_epoch{state.epoch}_step",
                                        keep=self.k + 1)
            heapq.heappush(self.checkpoints, (score, f"checkpoint_epoch{state.epoch}_step{state.step}"))

            # keep top-k, hence pop the smallest score
            if len(self.checkpoints) > self.k:
                _, worst_path = heapq.heappop(self.checkpoints)
                remove_dir_recursively(os.path.join(self.ckpt_dir, worst_path))

        if len(self.checkpoints) < self.k:

            save_checkpoint()

        elif score > self.checkpoints[-1][0]:

            assert len(self.checkpoints) == self.k
            save_checkpoint()

        else:

            assert len(self.checkpoints) == self.k
