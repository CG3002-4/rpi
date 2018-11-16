import numpy as np
import pandas as pd
from scipy import stats


FFT_NUM_AMPS = 3

def mean(sensors_data):
    return np.mean(sensors_data, axis=0)


def var(sensors_data):
    return np.var(sensors_data, axis=0)


def min(sensors_data):
    return np.min(sensors_data, axis=0)


def max(sensors_data):
    return np.max(sensors_data, axis=0)


def correlation(sensors_data):
    return np.hstack(
        [np.corrcoef(np.transpose(sensors_data[:, 0:3]))[[0, 0, 1], [1, 2, 2]],
         np.corrcoef(np.transpose(sensors_data[:, 3:6]))[[0, 0, 1], [1, 2, 2]],
         np.corrcoef(np.transpose(sensors_data[:, 6:9]))[[0, 0, 1], [1, 2, 2]],
         np.corrcoef(np.transpose(sensors_data[:, 9:12]))[[0, 0, 1], [1, 2, 2]]
         ]
    )

def fft(sensors_data):
    return np.abs(np.fft.rfft(sensors_data, axis=0))[1:]


def entropy(fft_vals):
    return stats.entropy(fft_vals, base=2)


def energy(fft_vals):
    np.sum(fft_vals ** 2, axis=0) / (fft_vals.shape[0] - 1)


def extract_features_over_segment(segment):
    sensors_data = segment.sensors_data
    means = mean(sensors_data)
    vars = var(sensors_data)
    mins = min(sensors_data)
    maxs = max(sensors_data)
    correlations = correlation(sensors_data)

    # fft_vals = fft(sensors_data)
    # entropies = entropy(fft_vals)
    # energies = energy(fft_vals)

    return np.hstack([means, vars, mins, maxs, correlations])


def extract_features(segments):
    return np.array([extract_features_over_segment(segment) for segment in segments])


if __name__ == '__main__':
    import os
    import data_collection
    import preprocess

    np.set_printoptions(suppress=True)

    EXP_LOCATION = os.path.join('data', 'varunchicken1')

    collector = data_collection.DataCollection(EXP_LOCATION)
    collector.load()

    segments = collector.segment()
    segments = preprocess.preprocess_segments(segments[0:3])
    print(extract_features(segments).shape)
