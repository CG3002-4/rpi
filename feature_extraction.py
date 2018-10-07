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


def extract_features_over_segment(segment, feature_extractors):
    """A feature extractor is a function that accepts a segment and
    returns a dataframe of features.
    """
    return pd.concat([feature_extractor(segment) for feature_extractor in feature_extractors], axis='columns')


def extract_features(segments, feature_extractors):
    """A feature extractor is a function that accepts a segment and
    returns a list of features.

    Features are extracted for each segment and returned as rows of a dataframe.
    """
    return pd.concat([extract_features_over_segment(segment, feature_extractors) for segment in segments], ignore_index=True)


if __name__ == '__main__':
    import data_collection
    import random

    np.set_printoptions(suppress=True)

    EXP_LOCATION = './data/test_exp/'

    collector = data_collection.DataCollection(EXP_LOCATION)
    collector.load()

    segments = collector.segment()
    print(extract_features(segments[0:3], [mean, stdev, correlate]))
