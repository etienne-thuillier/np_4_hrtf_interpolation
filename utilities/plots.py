import matplotlib.pyplot as plt
import numpy as np
import io
from utilities.utilities import flatten_dictionary

import logging


logger = logging.getLogger(__name__)


def make_metric_vs_sample_count_graph(sample_counts, metrics):

	metrics = flatten_dictionary(metrics)
	sample_counts = np.array(sample_counts)
	i = np.argsort(sample_counts)
	sample_counts = sample_counts[i]

	images = dict()
	for key, value in metrics.items():

		value = value[i]

		fig = plt.figure()

		axes = tuple(range(1, len(value.shape)))
		plt.plot(sample_counts, np.nanmean(value, axis=axes) + np.nanstd(value, axis=axes), ':b')
		plt.plot(sample_counts, np.nanmean(value, axis=axes), '-b')
		plt.plot(sample_counts, np.nanmean(value, axis=axes) - np.nanstd(value, axis=axes), ':b')

		plt.title(key)
		plt.xlabel('sample count')
		plt.xlim(0, sample_counts[-1])
		plt.grid()

		if 'relative error' in key:

			plt.ylim([None, 0])

			if 'complex' in key:

				plt.ylim([-30, 0])

		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		plt.close(fig)

		images = {key: plt.imread(buf, format='png'), **images}

	return images
