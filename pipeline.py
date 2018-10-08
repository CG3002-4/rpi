import sys
import numpy as np
import pickle
from data_collection import DataCollection
import feature_extraction
import train


FEATURE_EXTRACTORS = [feature_extraction.mean,
                      feature_extraction.stdev,
                      feature_extraction.correlate]


def pipeline(experiment_names):
    assert len(experiment_names) > 0

    segments = []

    for experiment_name in experiment_names:
        data_collection = DataCollection(experiment_name)
        data_collection.load()
        segments.extend(data_collection.segment())

    features = feature_extraction.extract_features(
        segments, FEATURE_EXTRACTORS)
    labels = np.array([segment.label for segment in segments])

    return features, labels


def train_pipeline(experiment_names, model_filename):
    features, labels = pipeline(experiment_names)

    model = train.train(X=features, y=labels)

    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)


def cross_validate_pipeline(experiment_names):
    features, labels = pipeline(experiment_names)

    train.cross_validate(X=features, y=labels)


if __name__ == '__main__':
    pipeline_type = sys.argv[1]

    if pipeline_type == 'train':
        train_pipeline(sys.argv[3:], sys.argv[2])
    elif pipeline_type == 'cross':
        cross_validate_pipeline(sys.argv[2:])
    else:
        print('Unrecognized pipeline: ' + pipeline_type)