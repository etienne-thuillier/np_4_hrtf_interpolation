import os, logging, random, pickle, scipy
import numpy as np
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from utilities.mps import mps
from utilities.utilities import register_resolvers
from utilities.coordinate_system import sph2cart__matlab
import sofar
from omegaconf import OmegaConf, DictConfig, ListConfig
import hydra
import torchaudio.functional as F


logger = logging.getLogger(__name__)


""" torch dataset """


def hutubs_files_from_plit(dataset_dir, split, basename_suffix):

    assert basename_suffix in ['simulated', 'measured']

    basenames = os.listdir(dataset_dir)
    basenames = [basename for basename in basenames if basename.split('_')[-1] == f'{basename_suffix}.sofa']

    train = ['pp10', 'pp11', 'pp12', 'pp13', 'pp14', 'pp15', 'pp16', 'pp17', 'pp19', 'pp2', 'pp20', 'pp21', 'pp22',
             'pp23', 'pp24', 'pp25', 'pp26', 'pp29', 'pp3', 'pp31', 'pp32', 'pp33', 'pp34', 'pp35', 'pp36', 'pp37',
             'pp38', 'pp39', 'pp40', 'pp41', 'pp42', 'pp43', 'pp44', 'pp45', 'pp46', 'pp47', 'pp48', 'pp49', 'pp5',
             'pp50', 'pp51', 'pp52', 'pp54', 'pp55', 'pp56', 'pp57', 'pp58', 'pp59', 'pp6', 'pp60', 'pp61', 'pp62',
             'pp63', 'pp64', 'pp66', 'pp68', 'pp69', 'pp7', 'pp70', 'pp71', 'pp72', 'pp73', 'pp74', 'pp75', 'pp76',
             'pp77', 'pp78', 'pp79', 'pp8', 'pp80', 'pp81', 'pp82', 'pp83', 'pp84', 'pp85', 'pp86', 'pp87', 'pp89',
             'pp9', 'pp90', 'pp91', 'pp92', 'pp93', 'pp94', 'pp95']
    test = ['pp28', 'pp30', 'pp4', 'pp53', 'pp65']
    valid = ['pp1', 'pp18', 'pp27', 'pp67']

    if split == 'all':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_')[0] in train + valid + test]
    elif split == 'train':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_')[0] in train]
    elif split == 'test':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_')[0] in test]
    elif split == 'valid':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_')[0] in valid]
    elif split in train + valid + test:
        logger.warning(f'single datum split: {split}')
        dataset_entries = [basename for basename in basenames if basename.split('_')[0] == split]
    else:
        raise ValueError

    return [os.path.join(dataset_dir, entry) for entry in dataset_entries]


def ari_files_from_plit(dataset_dir, split):

    basenames = os.listdir(dataset_dir)
    basenames = [basename for basename in basenames if basename.split('_nh')[0] == 'hrtf']

    # NB: nh10 and nh22 are excluded from set
    train = ['12', '154', '53', '162', '135', '65', '136', '70', '66', '89', '31', '5', '46',
             '11', '157', '52', '143', '13', '68', '138', '29', '164', '133', '63', '47', '4',
             '30', '71', '137', '160', '8', '161', '27', '42', '39', '58', '19', '35', '62',
             '74', '36', '61', '34', '18', '59', '38', '43', '14', '55', '54', '15', '146',
             '16', '41', '2', '40', '17', '56', '21', '60', '57', '140', '156']
    valid = ['32', '158', '159', '33', '64', '72']
    test = ['134', '25', '163', '28', '90', '139']

    if split == 'all':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_nh')[-1].split('.')[0] in train + valid + test]
    elif split == 'train':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_nh')[-1].split('.')[0] in train]
    elif split == 'test':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_nh')[-1].split('.')[0] in test]
    elif split == 'valid':
        dataset_entries = [basename for basename in basenames if
                           basename.split('_nh')[-1].split('.')[0] in valid]
    elif split in train + valid + test:
        logger.warning(f'single datum split: {split}')
        dataset_entries = [basename for basename in basenames if basename.split('_nh')[-1].split('.')[0] == split]
    else:
        raise ValueError

    return [os.path.join(dataset_dir, entry) for entry in dataset_entries]


def aachen_files_from_plit(dataset_dir, split):

    basenames = os.listdir(dataset_dir)
    basenames = [basename for basename in basenames if basename.split('.')[-1] == 'sofa']

    # NB: MRT02, MRT14 and MRT16 have been excluded from the dataset
    test = ['MRT17', 'MRT40', 'MRT07']
    valid = ['MRT06', 'MRT47']
    train = ['MRT10', 'MRT36', 'MRT20', 'MRT01', 'MRT21', 'MRT37', 'MRT41', 'MRT08', 'MRT24', 'MRT32', 'MRT27',
             'MRT31', 'MRT11', 'MRT46', 'MRT30', 'MRT26', 'MRT38', 'MRT33', 'MRT25', 'MRT09', 'MRT48', 'MRT29', 'MRT05',
             'MRT44', 'MRT13', 'MRT12', 'MRT45', 'MRT04', 'MRT28', 'MRT39', 'MRT03', 'MRT42', 'MRT15', 'MRT35', 'MRT23',
             'MRT19', 'MRT18', 'MRT22', 'MRT34', 'MRT43']

    if split == 'all':
        dataset_entries = [basename for basename in basenames if basename.split('.')[0] in train + valid + test]
    elif split == 'train':
        dataset_entries = [basename for basename in basenames if basename.split('.')[0] in train]
    elif split == 'test':
        dataset_entries = [basename for basename in basenames if basename.split('.')[0] in test]
    elif split == 'valid':
        dataset_entries = [basename for basename in basenames if basename.split('.')[0] in valid]
    elif split in train + valid + test:
        logger.warning(f'single datum split: {split}')
        dataset_entries = [basename for basename in basenames if basename.split('.')[0] == split]
    else:
        raise ValueError

    return [os.path.join(dataset_dir, entry) for entry in dataset_entries]


class SOFA(Dataset):

    def __init__(self, sofa_dir, sofa_sets, split, cardinality, transform, seed):

        super().__init__()

        self.seed = seed
        self.sofa_dir = sofa_dir
        self.dataset_entries = list()
        if 'hutubs_simulated' in sofa_sets:
            self.dataset_entries += hutubs_files_from_plit(dataset_dir=os.path.join(sofa_dir, 'hutubs'), split=split, basename_suffix='simulated')
        if 'hutubs_measured' in sofa_sets:
            self.dataset_entries += hutubs_files_from_plit(dataset_dir=os.path.join(sofa_dir, 'hutubs'), split=split, basename_suffix='measured')
        if 'ari' in sofa_sets:
            self.dataset_entries += ari_files_from_plit(dataset_dir=os.path.join(sofa_dir, 'ari'), split=split)
        if 'aachen' in sofa_sets:
            self.dataset_entries += aachen_files_from_plit(dataset_dir=os.path.join(sofa_dir, 'aachen'), split=split)

        self.subject_count = len(self.dataset_entries)

        if cardinality == None:
            cardinality = len(self.dataset_entries)

        repeat = 1
        if cardinality > len(self.dataset_entries):
            repeat = cardinality // len(self.dataset_entries)
            if cardinality % len(self.dataset_entries) > 0:
                logger.warning(f"Cardinality isn't a multiple of the number of dataset entries for split {split} of sets [{','.join(sofa_sets)}].")
                repeat += 1

        self.dataset_entries = self.dataset_entries * repeat

        if len(self.dataset_entries) > cardinality:
            logger.warning(f"Cardinality is smaller than (eventually repeated) set. Some subjects will appear in smaller proportion than others.")

        logger.warning(f'Truncating dataset to specified cardinality: {cardinality}.')
        self.dataset_entries = self.dataset_entries[:cardinality]

        assert len(self.dataset_entries) > 0, 'dataset is empty'
        self.transform = transform

    def __len__(self):
        return len(self.dataset_entries)

    def __getitem__(self, idx):

        datum = {'rng': np.random.default_rng(self.seed + idx),
                 'sofa_file': self.dataset_entries[idx],
                 'sofa_object': sofar.read_sofa(self.dataset_entries[idx], verify=False)}

        if self.transform:
            return self.transform(datum)
        else:
            return datum

    def get_subject_count(self):
        return self.subject_count


""" torch dataloader """


def collate(batch):

    out = list()
    for i in range(len(batch[0])):
        out.append([entry[i] for entry in batch])

    for i in range(len(batch[0])):
        if isinstance(out[i][0], str):
            pass
        else:
            out[i] = np.stack(out[i])

    return out


def worker_init_fn(worker_id):

    # following https://github.com/pytorch/pytorch/issues/5059#issuecomment-817275497
    worker_seed = (worker_id + torch.initial_seed()) % np.iinfo(np.int32).max

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_transform_chain_4_train_and_eval(target_sample_rate, n_taps, group_delay_frequency_range,
                                          statistics, spherical_integration_weights, p_permute_ears, observation_count,
                                          min_observation_count, max_observation_count, p_bernouilli_mask,
                                          uniform_s2_grids):

    return [to_cartesian_doa,
            to_unit_sphere,
            partial(resample, target_sample_rate=target_sample_rate),
            sofa_2_dict,
            partial(to_time_aligned_hrtf, sampling_rate=target_sample_rate,
                    n_taps=n_taps, group_delay_frequency_range=group_delay_frequency_range),
            drop_nyquist,
            to_real_valued,
            add_spin_dimension,
            partial(insert_dataset_statistics, statistics=statistics),
            partial(make_spherical_integration_weights,
                    spherical_integration_weights=spherical_integration_weights),
            # partial(truncate_sequence_dim, i_start=, i_end=)
            partial(randomly_permute_ears, p_permute_ears=p_permute_ears),
            partial(draw_mask, observation_count=observation_count,
                    min_observation_count=min_observation_count, max_observation_count=max_observation_count,
                    p_bernouilli_mask=p_bernouilli_mask, uniform_s2_grids=uniform_s2_grids),
            sort_mask,
            partial(dict_to_list, ordered_key_list=['mask', 'w', 'x', 'complex_envelope', 'mu_data', 'sigma_data'])]


def make_transform_chain_4_statistics(target_sample_rate, n_taps, group_delay_frequency_range):

    return [to_cartesian_doa,
            to_unit_sphere,
            partial(resample, target_sample_rate=target_sample_rate),
            sofa_2_dict,
            partial(to_time_aligned_hrtf, sampling_rate=target_sample_rate,
                    n_taps=n_taps, group_delay_frequency_range=group_delay_frequency_range),
            drop_nyquist,
            to_real_valued,
            add_spin_dimension,
            partial(dict_to_list, ordered_key_list=['x', 'complex_envelope'])]


def sofa_dataloader(sofa_dir, sofa_sets, split, cardinality, seed, transforms, batch_size,
                    num_workers, pin_memory, prefetch_factor, shuffle, drop_last, persistent_workers,
                    multiprocessing_context):

    assert isinstance(transforms, list)

    dataset = SOFA(sofa_dir=sofa_dir,
                   sofa_sets=sofa_sets,
                   split=split,
                   cardinality=cardinality,
                   seed=seed,
                   transform=Compose(transforms))

    if batch_size is None:
        batch_size = len(dataset)

    return DataLoader(dataset=dataset,
                      collate_fn=collate,
                      batch_size=batch_size,
                      num_workers=num_workers, pin_memory=pin_memory, timeout=0, prefetch_factor=prefetch_factor,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      worker_init_fn=worker_init_fn,
                      persistent_workers=persistent_workers,
                      multiprocessing_context=multiprocessing_context)


""" sofa object transforms """


def to_cartesian_doa(datum):
    datum['sofa_object'].SourcePosition = get_source_position_in_cartesian_coordinates_from_sofa(datum['sofa_object'])
    return datum


def to_unit_sphere(datum):
    r = np.sqrt(np.sum(datum['sofa_object'].SourcePosition ** 2, axis=-1, keepdims=True))
    datum['sofa_object'].SourcePosition /= r
    return datum


def resample(datum, target_sample_rate):
    if target_sample_rate != datum['sofa_object'].Data_SamplingRate:

        # kaiser_best in https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html
        datum['sofa_object'].Data_IR = F.resample(
            waveform=torch.tensor(datum['sofa_object'].Data_IR, device='cpu'),
            orig_freq=datum['sofa_object'].Data_SamplingRate,
            new_freq=target_sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        ).numpy(force=True)

    return datum


def sofa_2_dict(datum):
    return {'sofa_file': datum['sofa_file'],
            'rng': datum['rng'],
            'x': datum['sofa_object'].SourcePosition,
            'hrir': datum['sofa_object'].Data_IR}


""" dict transforms """


def to_time_aligned_hrtf(datum, sampling_rate, n_taps, group_delay_frequency_range):
    pure_delay, complex_envelope = to_time_aligned_hrtf_helper(h=datum['hrir'],
                                                               sampling_rate=sampling_rate,
                                                               n_taps=n_taps,
                                                               group_delay_frequency_range=group_delay_frequency_range)
    return {'pure_delay': pure_delay, 'complex_envelope': complex_envelope, **datum}


def to_real_valued(datum):

    def format(z):
        assert z.ndim == 3
        z = z.transpose(0, 2, 1)
        if z.dtype in [np.complex64, np.complex128]:
            z = np.stack((z.real, z.imag), axis=-1)
        else:
            z = np.expand_dims(z, axis=-1)
        return z

    datum['complex_envelope'] = format(datum['complex_envelope'])
    datum['pure_delay'] = format(datum['pure_delay'])
    datum['hrir'] = format(datum['hrir'])

    return datum


def add_spin_dimension(datum):

    datum['complex_envelope'] = np.expand_dims(datum['complex_envelope'], axis=-3)
    datum['pure_delay'] = np.expand_dims(datum['pure_delay'], axis=-3)
    datum['hrir'] = np.expand_dims(datum['hrir'], axis=-3)

    return datum


def insert_dataset_statistics(datum, statistics):
    return {'mu_data': statistics['mu_data'], 'sigma_data': statistics['sigma_data'], **datum}


def make_spherical_integration_weights(datum, spherical_integration_weights):
    assert datum['x'].ndim == 2, 'expecting coordinate of shape (datapoint, 3)'

    if spherical_integration_weights == 'identity':
        w = np.ones((datum['x'].shape[0],), dtype=datum['x'].dtype)
    elif spherical_integration_weights == 'spherical_voronoi_area':
        voronoi = scipy.spatial.SphericalVoronoi(points=datum['x'])
        w = voronoi.calculate_areas()
    else:
        raise ValueError

    # TODO: shouldn't we do w = w / w.sum() here?

    return {'w': w, **datum}


def randomly_permute_ears(datum, p_permute_ears):

    if datum['rng'].random() < p_permute_ears:

        datum['x'][..., 1] *= -1

        for key in [k for k in datum.keys() if k not in ['rng', 'x', 'sofa_file', 'w', 'mask']]:
            datum[key] = datum[key][..., [1, 0], :]

    return datum


def drop_nyquist(datum):
    if 'mu_data' in datum.keys():
        raise NotImplementedError('need to drop nyquist in the mean and std dev arrays as well...')
    assert datum['complex_envelope'].shape[-1] % 2 == 1, 'expecting odd number of bins for positive frequency side of spectrum'
    datum['complex_envelope'] = datum['complex_envelope'][..., :-1]
    return datum


def from_pickle_file(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def draw_mask(datum, observation_count, min_observation_count, max_observation_count, p_bernouilli_mask, uniform_s2_grids):

    max_observation_count_in_uniform_grid = ((uniform_s2_grids[-1] ** 2).sum(-1) > 1e-4).sum()

    if observation_count is None:
        assert max_observation_count is not None and min_observation_count is not None
        assert max_observation_count <= max_observation_count_in_uniform_grid
    else:
        assert observation_count <= max_observation_count_in_uniform_grid

    del max_observation_count_in_uniform_grid

    n_datapoints = datum['x'].shape[0]

    if datum['rng'].random() < p_bernouilli_mask:

        valid_mask = False
        while not valid_mask:

            p = observation_count / n_datapoints if observation_count is not None else None
            min_p = min_observation_count / n_datapoints if min_observation_count is not None else None
            max_p = max_observation_count / n_datapoints if max_observation_count is not None else None

            mask = draw_bernouilli_mask(rng=datum['rng'], x=datum['x'], x_weights=datum['w'], p=p, min_p=min_p, max_p=max_p,
                                        precision=1.0e-5, min_precision=None, max_precision=None)

            if np.all(mask.sum(-1) <= max_observation_count):
                valid_mask = True

    else:

        mask = draw_uniform_s2_grid(rng=datum['rng'], x=datum['x'], uniform_s2_grids=uniform_s2_grids,
                                    index=observation_count)
        mask = np.array(mask)

    return {'mask': mask, **datum}


def sort_mask(datum):
    """ Move observed datapoints to front of arrays.
        This reduces the space complexity when computing 2x2 kernel-based distances. """

    i = np.argsort(datum['mask'])
    i = np.flip(i)

    for key in [k for k in datum.keys() if k not in ['sofa_file', 'rng']]:
        datum[key] = datum[key][i]

    return datum


def dict_to_list(datum, ordered_key_list):
    return [datum[key] for key in ordered_key_list]


""" helpers """


def to_time_aligned_hrtf_helper(h, sampling_rate, n_taps, group_delay_frequency_range=[0, 1100]):
    from numpy.fft import fft as ft
    from numpy.fft import ifft as ift

    def get_group_delay(z, fftsize):
        """
        computes group delay https://ccrma.stanford.edu/~jos/fp/Group_Delay.html
            D(z, omega) = - d/domega angle(z)(omega)

        using finite differences

        returns group quantified as number of samples
        """
        phasor = np.roll(z, shift=-1) / z

        # plt.figure()
        # plt.plot(phasor[:, 0, :].real.transpose())
        # plt.plot(-phasor[:, 0, :].imag.transpose())

        delta_omega = 2 * np.pi / fftsize

        gp = - np.angle(phasor) / delta_omega

        return gp

    # minimum_phase_processor=partial(minimum_phase_jos, fftsize=256, mps_cutoff_dB=-40)
    def pure_delay_wegd(h, fftsize, frequencies, sampling_rate, minimum_phase_processor, axis=-1):
        """
        Weighted excess group delay (WEGD) method for extraction of pure delay component of hrirs according to

            Nam et al. - 2008 - A Method for Estimating Interaural Time Difference for Binaural Synthesis.

        Result is the power spectrum weighted average of the group delay over specified frequencies


        :param frequencies:                     None -> weighted excess group delay over the entire audio range
                                                [min freq. (Hz), max freq. (Hz)] -> weighted excess group delay over the
                                                specified frequency range
                                                trimming percentage -> weighted excess group delay over (100 - trimming)
                                                strongest magnitudes
        :param sampling_rate:                   sampling rate of h
        :minimum_phase_processor:               processor taking H=ft(h, fftsize) and returning minimum phase version
        :param axis:                            axis of filter taps in h
        :return:                                returns the pure delay (in samples)
         """

        permutation = list(range(len(h.shape)))
        permutation[-1] = axis
        permutation[axis] = len(h.shape) - 1

        h = h.transpose(permutation)

        # computing minimum phase (mp) component
        H = ft(h, n=fftsize, axis=-1)
        h_mp = minimum_phase_processor(h)
        H_mp = ft(h_mp, n=fftsize, axis=-1)

        # carrier = clipdb(H, cutoff_dB) / H_mp
        carrier = H / H_mp
        # import matplotlib.pyplot as plt
        #
        # # plt.plot(H_mp.real.transpose())
        # # plt.plot(H_mp.imag.transpose())
        # plt.plot(np.abs(H_mp[:, 0, :]).transpose())
        # plt.plot(carrier[:, 0, :].real.transpose())
        # # plt.plot(carrier.imag.transpose())

        group_delay = get_group_delay(carrier, fftsize)

        # # for debugging of group delay computation using scipy reference
        # # ---------------------------------------------------------------
        # import scipy
        # h_carrier = ift(carrier, axis=-1).real
        # _, gp_ref = scipy.signal.group_delay((h_carrier[0, 0, :], 1), w=fftsize, whole=True, fs=sampling_rate)
        # import matplotlib.pyplot as plt
        # plt.plot(20 * np.log10(np.abs((group_delay[0, 0, :] - gp_ref) / gp_ref)))

        # fig = plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(group_delay[:, 0, :].transpose())
        # plt.xlim([0, 100])
        # # plt.ylim([0, 0.5])
        # plt.subplot(2, 1, 2)
        # plt.plot(group_delay[:, 1, :].transpose())
        # plt.xlim([0, 100])
        # # plt.ylim([0, 0.5])
        # # fig.savefig('/tmp/tmp/plot_%i.png' % count, format='png', dpi=150)
        # # plt.close(fig)

        # import matplotlib
        # fig = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(matplotlib.colors.Normalize(vmin=0, vmax=0.5, clip=True)(group_delay[:, 0, :100]))
        # # plt.colorbar()
        # plt.subplot(1, 2, 2)
        # plt.imshow(matplotlib.colors.Normalize(vmin=0, vmax=0.5, clip=True)(group_delay[:, 1, :100]))
        # # plt.colorbar()
        # # fig.savefig('/tmp/tmp/imshow_%i.png' % count, format='png', dpi=150)
        # # plt.close(fig)
        # # count += 1

        # # plt.ylim((0, 0.5))
        # # import jax.numpy as jnp

        # norm = matplotlib.colors.Normalize(vmin=0, vmax=0.05, clip=True)
        # plt.imshow(norm(np.sort(group_delay[:, 0, :])))
        # plt.colorbar()
        # pure_delay_frequency_cutoff = 500

        # weighted average of group delay in lower part of spectrum
        if isinstance(frequencies, float):
            # trimming case
            assert frequencies > 0 and frequencies < 1
            assert False, 'not yet coded'

        elif isinstance(frequencies, tuple) or isinstance(frequencies, list) or isinstance(frequencies, ListConfig):
            # frequency range case
            n0 = int(frequencies[0] * fftsize / sampling_rate)
            nf = int(frequencies[1] * fftsize / sampling_rate)
            pure_delay = (group_delay[..., n0:nf + 1] * np.abs(H[..., n0:nf + 1]) ** 2).sum(-1, keepdims=True) / \
                         np.abs(np.abs(H[..., n0:nf + 1]) ** 2).sum(-1, keepdims=True)

        else:
            assert frequencies is None
            # all frequencies
            pure_delay = (group_delay * np.abs(H ** 2)).sum(-1, keepdims=True) / np.abs(np.abs(H) ** 2).sum(-1,
                                                                                                            keepdims=True)

        return pure_delay.transpose(permutation)

    def minimum_phase_jos(h, fftsize, mps_cutoff_dB, discard_second_half=False):
        """
        :param h:                               set of hrir
        :param mps_cutoff_dB:                   parameter of JOS's minimum phase spectrum function
        :param n_taps:                          length of taps of hrir that yielded H (might be differenct from fftsize)
        :return:                                set of minimum phase HRTF spectra, with shape (..., frequency bins)
        """
        H = ft(h, n=fftsize, axis=-1)
        H_mp = mps(H, cutoff_dB=mps_cutoff_dB, axis=-1)
        h_mp = ift(H_mp, axis=-1).real
        if discard_second_half:
            return h_mp[..., :h.shape[-1] // 2]
        else:
            return h_mp[..., :h.shape[-1]]

    def demodulate(h, pure_delay, sample_rate, fftsize):
        """
        :param h:                           hrir set with shape (..., filter taps)
        :param pure_delay:                  pure delay in samples for each hrir filter
        :param sample_rate:                 sample rate of hrir filters
        :param fftsize:                     int
        :return:                            complex spectrum envelope
        """

        if pure_delay.shape == h.shape:

            def make_demodulating_phasor():
                cum_tau = np.cumsum(np.concatenate((np.zeros((*pure_delay.shape[:-1], 1), dtype=pure_delay.dtype),
                                                    pure_delay[..., 1:fftsize // 2 + 1]), axis=-1), axis=-1)
                return np.exp(2j * np.pi * cum_tau * sample_rate / fftsize)

        else:

            def make_demodulating_phasor():
                omegas = 2 * np.pi * np.arange(fftsize // 2 + 1) / fftsize
                return np.exp(1j * omegas * pure_delay)

        spectrum_envelope = ft(h, n=fftsize)[..., :fftsize // 2 + 1] * make_demodulating_phasor()

        return spectrum_envelope

    assert n_taps >= h.shape[-1], 'Expecting n_taps to introduce zeropadding, here it would truncate the hrir...'

    pure_delay = pure_delay_wegd(h=h,
                                 fftsize=n_taps,
                                 frequencies=group_delay_frequency_range,
                                 sampling_rate=sampling_rate, #44100 * fftsize / 256,
                                 minimum_phase_processor=partial(minimum_phase_jos,
                                                                 fftsize=n_taps * 8,
                                                                 mps_cutoff_dB=-40),
                                 axis=-1)

    complex_envelope = demodulate(h=h,
                                  pure_delay=pure_delay,
                                  sample_rate=sampling_rate,
                                  fftsize=n_taps)

    return pure_delay, complex_envelope


def get_source_position_in_cartesian_coordinates_from_sofa(sofa_object):

    if sofa_object.SourcePosition_Type == 'cartesian' and sofa_object.SourcePosition_Units == 'meter':

        x1, x2, x3 = sofa_object.SourcePosition.transpose()

        # assert np.allclose(x1 ** 2 + x2 ** 2 + x3 ** 2, 1), 'expect hrtf measured at location on sphere of unit radius'
        x_out = np.stack((x1, x2, x3), axis=-1)

    elif sofa_object.SourcePosition_Type == 'spherical' and sofa_object.SourcePosition_Units == 'degree, degree, metre':

        azimuth, elevation, radius = sofa_object.SourcePosition.transpose()

        assert np.all(radius >= 1), 'expect hrtf measured at location on sphere of at least unit radius'

        x_out = sph2cart__matlab(azimuth=np.deg2rad(azimuth), elevation=np.deg2rad(elevation), radius=radius)
        x_out = np.stack(x_out, axis=-1)

    return x_out


def draw_bernouilli_mask(rng, x, x_weights, p, min_p, max_p, precision, min_precision, max_precision):
    """ Draw context mask with shape x.shape[:-1] using a spherical Gaussian kernel.

        p:          percentage of coordinates which mask is drawn with positive value
    """

    ''' randomly draw Bernoulli parameter if unspecified '''
    if p is None:
        assert min_p is not None and max_p is not None
        p = rng.uniform(min_p, max_p)
    else:
        assert min_p is None and max_p is None

    ''' randomly draw mode_scale if unspecified '''
    if precision is None:
        assert min_precision is not None and max_precision is not None
        precision = rng.uniform(min_precision, max_precision)
    else:
        assert min_precision is None and max_precision is None

    ''' mode location '''
    mu = rng.normal(size=(3,))
    mu /= np.linalg.norm(mu)

    ''' probability density of spherical Gaussian kernel at 3-d coordinates '''
    assert np.allclose((x ** 2).sum(-1), 1.0)
    cos_theta = np.clip(np.sum(x * mu, axis=-1), -1.0, 1.0)
    q = x_weights * np.exp(precision * (cos_theta - 1))

    ''' drawing probability at each 3-d coordinate '''
    p_x = p * np.prod(x.shape[:-1]) * q / q.sum()

    ''' draw mask '''
    mask = rng.random(x.shape[:-1]) < p_x

    return mask


def make_rotation_matrix(rotation_axis, theta):
    """
    adapted from
        https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

    follows from
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation

    :param rotation_axis:   unit vector on S^2, defines axis of rotation
    :param theta:           rotation angle, in [0, 2 pi]
    :return:                rotation matrix R, which can be applied to vector following dot(M0, v)
    """

    rotation_axis /= np.sqrt((rotation_axis ** 2).sum(keepdims=True))
    R = scipy.linalg.expm(np.cross(np.eye(3), rotation_axis * theta))

    return R


def draw_uniform_s2_grid(rng, x, uniform_s2_grids, index=None):

    def random_rotation(x_c):

        rotation_axis = rng.normal(size=(3,))
        rotation_axis /= np.sqrt((rotation_axis ** 2).sum())

        theta = 2 * np.pi * rng.uniform()

        R = make_rotation_matrix(rotation_axis, theta)

        return np.einsum('ij,kj->ik', x_c, R)

    def flip_bit_of_candidate_closest_to_target(x_candidate, x_target):

        closeness = np.einsum('ji,i->j', x_candidate, x_target)

        max_mask = closeness == np.max(closeness)

        x_target_is_on_s2 = (x_target ** 2).sum() != 0  # discard targets with 0 modulus (not on s2)
        # (we have some here by design to avoid indexing within jitted
        #  functions)
        return max_mask * x_target_is_on_s2

    if index is None:

        x_c = rng.choice(uniform_s2_grids, (1,))[0]

    else:

        x_c = uniform_s2_grids[index]

    x_c = random_rotation(x_c)

    mask = np.zeros(shape=(x.shape[0],), dtype=bool)
    for i in range(x_c.shape[0]):
        mask += flip_bit_of_candidate_closest_to_target(x, x_c[i])
    assert (np.abs(x_c[:,0])>0).sum() == mask.sum()

    return mask


@hydra.main(version_base=None, config_path="conf", config_name="train_sofa_loader")
def demo(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    dataloader = hydra.utils.instantiate(cfg.data.train)

    for i, (mask, w, x, complex_envelope, mu_data, sigma_data) in enumerate(dataloader):

        print(f"batch {i}")
        # print(f"sofa_file: {len(sofa_file)}")
        print(f"mask: {mask.shape}")
        print(f"w: {w.shape}")
        print(f"x: {x.shape}")
        print(f"complex_envelope: {complex_envelope.shape}")
        print(f"mu_data: {mu_data.shape}")
        print(f"sigma_data: {sigma_data.shape}")


if __name__ == "__main__":

    register_resolvers()

    demo()
