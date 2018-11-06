import numpy as np
import processed_sensor_data as psd
import segment as seg
from scipy import signal
from data_collection import NUM_SENSORS
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


SAMPLING_FREQ = 60
NYQ_FREQ = SAMPLING_FREQ * 0.5


def medfilt(axis):
    return signal.medfilt(axis)


SOS = signal.butter(3, 20.0 / NYQ_FREQ, btype='lowpass', output='sos')
GRAV_FILTER = signal.butter(3, 20.0 / NYQ_FREQ, btype='lowpass', output='sos')
BODY_FILTER = signal.butter(3, 20.0 / NYQ_FREQ, btype='highpass', output='sos')


def butter_grav_body_sep(axis, filter_type):
    cutoff_freq = 0.3 / NYQ_FREQ
    order = 3
    sos = signal.butter(order, cutoff_freq, btype=filter_type, output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def butter_gravity(axis):
    return signal.sosfiltfilt(GRAV_FILTER, axis, padlen=0)


def butter_body(axis):
    return signal.sosfiltfilt(BODY_FILTER, axis, padlen=0)


def filter_data(filter, data):
    return np.apply_along_axis(filter, 0, data)


def preprocess_segment(segment):
    """Takes in a segment and returns it with processed sensor data.

    Each segment is first noise-filtered then acc data is split into
    bodyAcc and gravAcc data.

    The noise filters are applied in left to right order.
    """
    sensors_data = np.apply_along_axis(func1d=signal.medfilt, arr=segment.sensors_data, axis=0)
    # new_data = np.empty((sensors_data.shape[0], 18))
    # sensors_data = np.apply_along_axis(func1d=lambda axis: signal.sosfiltfilt(SOS, axis, padlen=0), arr=segment.sensors_data, axis=0)
    # new_data[:, :6] = np.apply_along_axis(func1d=butter_body, arr=sensors_data[:, [0, 1, 2, 6, 7, 8]], axis=0)
    # new_data[:, 6:12] = np.apply_along_axis(func1d=butter_gravity, arr=sensors_data[:, [0, 1, 2, 6, 7, 8]], axis=0)
    # new_data[:, 12:] = sensors_data[:, [3, 4, 5, 9, 10, 11]]
    segment.sensors_data = sensors_data
    return segment


def preprocess_segments(segments):
    """Takes in a list of segments and returns a new list of segments that
    contain processed sensor data.
    """
    return [preprocess_segment(segment) for segment in segments]


if __name__ == '__main__':
    import os
    import data_collection

    np.set_printoptions(suppress=True)

    EXP_LOCATION = os.path.join('data', 'varunchicken1')

    collector = data_collection.DataCollection(EXP_LOCATION)
    collector.load()

    segments = collector.segment()
    processed_segments = (preprocess_segments(segments[0:3]))

    # import matplotlib.pyplot as plt
    # import plot
    #
    # plt.figure(facecolor="white", figsize=(15, 10))
    #
    # plt.subplot(131)
    # plot.plot_data(segments[0].sensors_data[0].acc, 'Acc')
    #
    # plt.subplot(132)
    # filtered_acc = filter_data(
    #     butter_noise,
    #     filter_data(
    #         medfilt,
    #         segments[0].sensors_data[0].acc
    #     )
    # )
    # plot.plot_data(filtered_acc, 'Filtered_acc')
    #
    # plt.subplot(133)
    # body_plus_grav = (processed_segments[0].sensors_data[0].body +
    #                   processed_segments[0].sensors_data[0].grav)
    # plot.plot_data(body_plus_grav, 'Body_plus_grav')
    #
    # plt.show()
