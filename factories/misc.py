from functools import partial

import hydra
import optax

import callbacks as cb
from callbacks import evaluate_uncertainty_miscalibration__helper
from metrics import reduced_per_feature_metric, per_feature_negative_log_likelihood, \
    per_feature_relative_error_db, per_feature_log_mag_distance_db


def make_optimizer_setup(n_epochs, steps_per_epoch, learning_rate_type, learning_rate, weight_decay, decay_rate=None):
    if learning_rate_type == 'optimizer_hyperparameter':

        assert decay_rate is None, 'Learning rate''s decay rate is set despite learning_rate_type == ''optimizer_hyperparameter'', i.e. not using a scheduler. Something is fishy with the configuration.'
        return dict(optimizer_hyperparameters=dict(learning_rate=learning_rate),
                    optimizer_factory=partial(optax.adamw,
                                              weight_decay=weight_decay))

    elif learning_rate_type == 'scheduled':
        assert decay_rate is not None

        learning_rate = optax.exponential_decay(init_value=learning_rate,
                                                transition_steps=n_epochs * steps_per_epoch,
                                                decay_rate=decay_rate)

        return dict(optimizer_hyperparameters=dict(),
                    optimizer_factory=partial(optax.adamw,
                                              learning_rate=learning_rate,
                                              weight_decay=weight_decay))
    else:
        raise ValueError


def make_metrics(metrics, exclude_context_points_from_metrics, metric_reduction, **kwargs):
    return {key: partial(reduced_per_feature_metric,
                         per_feature_metric=value['per_feature_metric'],
                         exclude_context_points=exclude_context_points_from_metrics,
                         reduction_mode=metric_reduction,
                         sample_reduction_mode=value['sample_reduction_mode'])
            for key, value in
            {'per-feature negative log-likelihood': dict(per_feature_metric=per_feature_negative_log_likelihood,
                                                         sample_reduction_mode='Monte-Carlo: negative log-likelihood'),
             'per-feature relative error (dB)': dict(per_feature_metric=per_feature_relative_error_db,
                                                     sample_reduction_mode='mean'),
             'per-feature log-mag distance (dB)': dict(per_feature_metric=per_feature_log_mag_distance_db,
                                                       sample_reduction_mode='mean')}.items()
            if key in metrics}


def make_callbacks(cfg, eval_key, metrics, writer, statistics, split):
    transform_chain_factory = hydra.utils.instantiate(cfg.data.transforms)
    transform_chain_factory = partial(transform_chain_factory,
                                      statistics=statistics,
                                      observation_count=None,
                                      min_observation_count=0,
                                      max_observation_count=cfg.data.max_observation_count,
                                      p_permute_ears=0.5,
                                      p_bernouilli_mask=0.5)

    dataloader_factory = hydra.utils.instantiate(cfg.data.dataloader_factory)
    loader_factory = partial(dataloader_factory, split=split, **cfg.data.splits[split])

    callbacks = {}

    if 'evaluation' in cfg.callbacks.keys():

        if 'checkpoint_manager' in cfg.callbacks.evaluation.keys():
            checkpoint_manager = hydra.utils.instantiate(cfg.callbacks.evaluation.checkpoint_manager)
        else:
            checkpoint_manager = None

        transforms = transform_chain_factory(p_bernouilli_mask=0.0)
        eval_iter = loader_factory(transforms=transforms)

        callbacks.update({'evaluation': partial(cb.filter_callback,
                                                callback=partial(cb.log_evaluation,
                                                                 rng=eval_key,
                                                                 writer=writer,
                                                                 eval_iter=eval_iter,
                                                                 L=1,
                                                                 metrics=metrics,
                                                                 checkpoint_manager=checkpoint_manager),
                                                period=cfg.callbacks['evaluation'].period)})

    if 'metric vs sample count' in cfg.callbacks.keys():
        transforms = partial(transform_chain_factory,
                             min_observation_count=None,
                             max_observation_count=None,
                             p_permute_ears=0.5,
                             p_bernouilli_mask=0.0)

        count_2_iterable = {f'{count}': loader_factory(transforms=transforms(observation_count=count))
                            for count in range(cfg.callbacks['metric vs sample count'].sample_count_step,
                                               cfg.data.max_observation_count + 1,
                                               cfg.callbacks['metric vs sample count'].sample_count_step)}

        callback = partial(cb.filter_callback,
                           callback=partial(cb.plot_metrics_vs_sample_count,
                                            key=eval_key,
                                            L=1,
                                            metrics=metrics,
                                            count_2_dataset_iterables=count_2_iterable,
                                            images_writer=writer),
                           period=cfg.callbacks['metric vs sample count'].period)

        callbacks.update({'metric vs sample count': callback})

    if 'plot examples' in cfg.callbacks.keys():
        per_feature_metrics = hydra.utils.instantiate(cfg.metrics.per_feature_metrics)

        transforms = partial(transform_chain_factory,
                             min_p=None,
                             max_p=None,
                             p_permute_ears=0.5,
                             p_bernouilli_mask=0.0)

        count_2_iterable = {f'{count}': loader_factory(transforms=transforms(p=count / 1730))
                            for count in cfg.callbacks['plot examples'].sample_counts}

        raise NotImplemented
        make_plot_job = partial(make_qualitative_example_plot_jobs__time_aligned_hrtf,
                                latent_x=None,
                                peak_magnitude=statistics['peak_magnitude'],
                                max_dimension_size=8,
                                datum_transform='identity',
                                input_preprocessing=cfg.model.model_factory.input_preprocessing,
                                plot_data_statistics=True,
                                config_string='dummy',
                                input_features=INPUT_FEATURES)

        callback = partial(cb.filter_callback,
                           callback=partial(cb.plot_examples,
                                            key=eval_key,
                                            dataset_iterable=count_2_iterable,
                                            L=1,
                                            image_writer=writer,
                                            per_feature_metrics=per_feature_metrics,
                                            make_plot_job=make_plot_job,
                                            multiple_samples_per_plot=False),
                           period=cfg.callbacks['plot examples'].period)

        callbacks.update({'plot examples': callback})

    if 'evaluate uncertainty miscalibration' in cfg.callbacks.keys():
        transforms = transform_chain_factory(p_bernouilli_mask=0.0)
        callback = partial(cb.filter_callback,
                           callback=partial(evaluate_uncertainty_miscalibration__helper,
                                            key=eval_key,
                                            writer=writer,
                                            eval_iter=loader_factory(transforms=transforms),
                                            n_bins=2 ** 5,
                                            feature_decimating_factors=cfg.callbacks[
                                                'evaluate uncertainty miscalibration'].feature_decimating_factors,
                                            eval_batch=cfg.data.splits.valid.batch_size),
                           period=cfg.callbacks['evaluate uncertainty miscalibration'].period)
        callbacks.update({'evaluate uncertainty miscalibration': callback})

    if 'plot input set convolution kernel parameters' in cfg.callbacks.keys():
        callback = partial(cb.filter_callback,
                           callback=partial(cb.plot_input_set_convolution_kernel_array_parameters,
                                            writer=writer,
                                            t0=0,
                                            n_sequence=96),
                           period=cfg.callbacks['plot input set convolution kernel parameters'].period)
        callbacks.update({'plot input set convolution kernel'' parameters': callback})

    return callbacks
