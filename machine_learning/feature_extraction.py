import numpy as np


def extract_feature_over_each_sensor(segment, feature_extractor_over_sensor):
    """A feature extractor over sensor is a function that accepts a
    sensor_data.SensorData and returns a list of features.
    """
    return np.concatenate([
        feature_extractor_over_sensor(sensor_data)
        for sensor_data in segment.sensors_data
    ])


def extract_feature_over_all_axes(segment, feature_extractor_over_axis):
    """A feature extractor over axis is a function that accepts a 1D array
    and returns a list of features.
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


# def correlate(segment):



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
    print(extract_features(segments[0], [mean, stdev]))
