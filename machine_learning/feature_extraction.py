import numpy as np


def extract_feature_over_each_sensor(segment, feature_extractor_over_sensor):
    """A feature extractor over sensor is a function that accepts a
    sensor_data.SensorData and returns a list of features.

    Such an extractor is applied to each sensor to obtain a list of features.
    """
    return np.concatenate([
        feature_extractor_over_sensor(sensor_data)
        for sensor_data in segment.sensors_data
    ])


def extract_feature_over_all_axes(segment, feature_extractor_over_axis):
    """A feature extractor over axis is a function that accepts a 1D array
    and returns a list of features.

    Such an extractor is applied to each axis in each sensor to obtain a list
    of features.
    """
    return extract_feature_over_each_sensor(
        segment,
        lambda sensor_data:
            np.apply_along_axis(
                feature_extractor_over_axis,
                arr=sensor_data.get_all_axes(),
                axis=0
            )
    )


def mean(segment):
    return extract_feature_over_all_axes(segment, np.mean)


def stdev(segment):
    return extract_feature_over_all_axes(segment, np.std)


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

    return extract_feature_over_each_sensor(segment, correlate_over_sensor)


def extract_features(segment, feature_extractors):
    """A feature extractor is a function that accepts a segment and
    returns a list of features.
    """
    return np.concatenate([feature_extractor(segment) for feature_extractor in feature_extractors])


if __name__ == '__main__':
    import data_collection
    import random

    DATA_FILE = 'collection_test.pb'

    collector = data_collection.DataCollection(DATA_FILE)
    collector.load()

    labels = [random.randrange(1, 12) for i in range(len(collector.move_start_indices))]

    segments = collector.segment(labels)
    print(extract_features(segments[0], [mean, stdev, correlate]))
