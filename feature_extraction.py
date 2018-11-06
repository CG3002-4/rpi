import numpy as np
import pandas as pd


def extract_feature_over_each_sensor(segment, feature_extractor_over_sensor, feature_names):
    """A feature extractor over sensor is a function that accepts a
    sensor_data.SensorData and returns a list of features or a single feature.

    Such an extractor is applied to each sensor to obtain a dataframe of features.
    """
    feature_names = ['Sensor' + str(i) + '_' + feature_name
                     for i in range(len(segment.sensors_data))
                     for feature_name in feature_names
                     ]

    features = np.concatenate([
        feature_extractor_over_sensor(sensor_data)
        for sensor_data in segment.sensors_data
    ])

    # Need to convert numpy array into shape (1, num_features)
    return pd.DataFrame(data=np.atleast_2d(features), columns=feature_names)


def extract_feature_over_all_axes(segment, feature_extractor_over_axis, feature_name):
    """A feature extractor over axis is a function that accepts a 1D array
    and returns a list of features or a single feature.

    Such an extractor is applied to each axis in each sensor to obtain a dataframe
    of features.
    """
    feature_names = [feature_name + '_' + sensor_type + '_' + axis_name
                     for sensor_type in ['Acc', 'Gyro']
                     for axis_name in ['x', 'y', 'z']
                     ]

    return extract_feature_over_each_sensor(
        segment,
        lambda sensor_data:
            np.apply_along_axis(
                feature_extractor_over_axis,
                arr=sensor_data.get_all_axes(),
                axis=0
            ),
        feature_names
    )


def max(segment):
    return extract_feature_over_all_axes(segment, np.max, 'Max')


def min(segment):
    return extract_feature_over_all_axes(segment, np.min, 'Min')


def var(segment):
    return extract_feature_over_all_axes(segment, np.var, 'Var')


def mean(segment):
    return extract_feature_over_all_axes(segment, np.mean, 'Mean')


def stdev(segment):
    return extract_feature_over_all_axes(segment, np.std, 'Stdev')


def correlate(segment):
    def correlate_over_sensor(sensor_data):
        def correlate_over_triaxial_data(triaxial_data):
            # We want Cxy, Cxz and Cyz
            # These are present at locations (0, 1), (0, 2) and (1, 2) respectively
            # To obtain these values using numpy, we need to provide a list of all
            # the row indices and a list of all the column indices
            return np.corrcoef(np.transpose(triaxial_data))[[0, 0, 1], [1, 2, 2]]

        return np.concatenate([
            correlate_over_triaxial_data(sensor_data.acc),
            correlate_over_triaxial_data(sensor_data.gyro)
        ])

    return extract_feature_over_each_sensor(
        segment,
        correlate_over_sensor,
        ['Corr' + '_' + sensor_type + '_' + axis_pair
         for sensor_type in ['Acc', 'Gyro']
         for axis_pair in ['xy', 'xz', 'yz']
         ]
    )

FFT_NUM_AMPS = 3
def extract_features_over_segment(segment):
    """A feature extractor is a function that accepts a segment and
    returns a dataframe of features.
    """
    sensors_data = segment.sensors_data
    means = np.mean(sensors_data, axis=0)
    vars = np.var(sensors_data, axis=0)
    mins = np.min(sensors_data, axis=0)
    maxs = np.max(sensors_data, axis=0)
    corrs = np.hstack(
        [np.corrcoef(np.transpose(sensors_data[:, 0:3]))[[0, 0, 1], [1, 2, 2]],
         np.corrcoef(np.transpose(sensors_data[:, 3:6]))[[0, 0, 1], [1, 2, 2]],
         np.corrcoef(np.transpose(sensors_data[:, 6:9]))[[0, 0, 1], [1, 2, 2]],
         np.corrcoef(np.transpose(sensors_data[:, 9:12]))[[0, 0, 1], [1, 2, 2]]
         # np.corrcoef(np.transpose(sensors_data[:, 12:15]))[[0, 0, 1], [1, 2, 2]],
         # np.corrcoef(np.transpose(sensors_data[:, 15:18]))[[0, 0, 1], [1, 2, 2]]
         ]
    )
    rfft = np.abs(np.fft.rfft(sensors_data, axis=0))
    # freq_amps = np.reshape(rfft[:FFT_NUM_AMPS, :], (-1,))
    energy = np.sum(rfft[1:] ** 2, axis=0) / (rfft.shape[0] - 1)
    return np.hstack([means, vars, mins, maxs, corrs, energy])


def extract_features(segments):
    """A feature extractor is a function that accepts a segment and
    returns a list of features.

    Features are extracted for each segment and returned as rows of a dataframe.
    """
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
