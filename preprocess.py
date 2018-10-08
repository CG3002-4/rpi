import numpy as np
import processed_sensor_data as psd
import segment as seg
from scipy import signal
from data_collection import NUM_SENSORS


SAMPLING_FREQ = 50
NYQ_FREQ = SAMPLING_FREQ * 0.5


def medfilt(axis):
    return signal.medfilt(axis)


def butter_noise(axis):
    cutoff_freq = 20.0 / NYQ_FREQ
    order = 3
    sos = signal.butter(order, cutoff_freq, btype='lowpass', output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def butter_grav_body_sep(axis, filter_type):
    cutoff_freq = 0.3 / NYQ_FREQ
    order = 3
    sos = signal.butter(order, cutoff_freq, btype=filter_type, output='sos')
    return signal.sosfiltfilt(sos, axis, padlen=0)


def butter_gravity(axis):
    return butter_grav_body_sep(axis, filter_type='lowpass')


def butter_body(axis):
    return butter_grav_body_sep(axis, filter_type='highpass')


def filter_data(filter, data):
    return np.apply_along_axis(filter, 0, data)


def preprocess_segment(segment, noise_filters):
    """Takes in a segment and returns it with processed sensor data.

    Each segment is first noise-filtered then acc data is split into
    bodyAcc and gravAcc data.

    The noise filters are applied in left to right order.
    """
    processed_sensors_data = []

    for i in range(NUM_SENSORS):
        acc = segment.sensors_data[i].acc
        gyro = segment.sensors_data[i].acc

        for noise_filter in noise_filters:
            acc = filter_data(noise_filter, acc)
            gyro = filter_data(noise_filter, gyro)

        body = filter_data(butter_body, acc)
        grav = filter_data(butter_gravity, acc)

        processed_sensors_data.append(psd.ProcessedSensorData(
            body_values=body,
            grav_values=grav,
            gyro_values=gyro
        ))

    return seg.Segment(sensors_data=processed_sensors_data, label=segment.label)


def preprocess_segments(segments, noise_filters):
    """Takes in a list of segments and returns a new list of segments that
    contain processed sensor data.
    """
    return [preprocess_segment(segment, noise_filters) for segment in segments]


if __name__ == '__main__':
    import os
    import data_collection

    np.set_printoptions(suppress=True)

    EXP_LOCATION = os.path.join('data', 'test_exp')

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
    filtered_acc = filter_data(
        butter_noise,
        filter_data(
            medfilt,
            segments[0].sensors_data[0].acc
        )
    )
    plot.plot_data(filtered_acc, 'Filtered_acc')

    plt.subplot(133)
    body_plus_grav = (processed_segments[0].sensors_data[0].body +
                      processed_segments[0].sensors_data[0].grav)
    plot.plot_data(body_plus_grav, 'Body_plus_grav')

    plt.show()
