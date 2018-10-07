import numpy as np
import pandas as pd
import processed_sensor_data as psd
import segment as seg
import copy
from scipy import stats, signal, fftpack
from data_collection import NUM_SENSORS


SAMPLING_FREQ = 50


def medfilt(axis):
    return signal.medfilt(axis)


def butter_noise(axis):
    nyq_freq = SAMPLING_FREQ * 0.5
    cutoff_freq = 20.0 / nyq_freq
    order = 3
    sos = signal.butter(order, cutoff_freq, btype='lowpass', output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def butter_gravity(axis):
    nyq_freq = SAMPLING_FREQ * 0.5
    cutoff_freq = 0.3 / nyq_freq
    order = 3
    sos = signal.butter(order, cutoff_freq, btype='lowpass', output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def butter_body(axis):
    nyq_freq = SAMPLING_FREQ * 0.5
    cutoff_freq = 0.3 / nyq_freq
    order = 3
    sos = signal.butter(order, cutoff_freq, btype='highpass', output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def filter_axis(filt, arr):
    return np.apply_along_axis(filt, 0, arr)


def preprocess_segment(segment, noise_filters):
    """
    Takes in a segment and returns it with processed sensor data.

    Each segment is first noise-filtered then acc data is split into
    bodyAcc and gravAcc data.
    """

    processed_segment = copy.deepcopy(segment)
    for i in range(NUM_SENSORS):
        processed_segment.sensors_data[i] = psd.ProcessedData(
            processed_segment.sensors_data[i].acc, processed_segment.sensors_data[i].acc, processed_segment.sensors_data[i].gyro)

        for noise_filter in noise_filters:
            processed_segment.sensors_data[i].body = filter_axis(
                noise_filter, processed_segment.sensors_data[i].body)
            processed_segment.sensors_data[i].grav = filter_axis(
                noise_filter, processed_segment.sensors_data[i].grav)
            processed_segment.sensors_data[i].gyro = filter_axis(
                noise_filter, processed_segment.sensors_data[i].gyro)

        processed_segment.sensors_data[i].body = filter_axis(
            butter_body, processed_segment.sensors_data[i].body)
        processed_segment.sensors_data[i].grav = filter_axis(
            butter_gravity, processed_segment.sensors_data[i].grav)

    return processed_segment


def preprocess_segments(segments, noise_filters):
    """
    Takes in a list of segments and
    returns a new list of segments that contain processed sensor data.
    """
    return [preprocess_segment(segment, noise_filters) for segment in segments]


if __name__ == '__main__':
    import data_collection

    np.set_printoptions(suppress=True)

    EXP_LOCATION = './data/test_exp/'

    collector = data_collection.DataCollection(EXP_LOCATION)
    collector.load()

    segments = collector.segment()
    processed_segments = (preprocess_segments(
        segments[0:3], [medfilt, butter_noise]))

    import matplotlib.pyplot as plt
    import plot

    plt.figure(facecolor="white", figsize=(15, 10))

    plt.subplot(131)
    plot.plot_data(segments[0].sensors_data[0].acc, 'Acc')

    plt.subplot(132)
    plot.plot_data(processed_segments[0].sensors_data[0].body, 'Body')

    plt.subplot(133)
    plot.plot_data(processed_segments[0].sensors_data[0].grav, 'Grav')

    plt.show()
