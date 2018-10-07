"""Implements the training and validation pipelines.

Usage:
    training:
        python3 pipeline train <model_filename> [<experiment_name>]
    cross validation:
        python3 pipeline cross [<experiment_name>]
"""
import sys
import numpy as np
import pickle
from machine_learning.data_collection import DataCollection
from machine_learning import feature_extraction
from machine_learning import train


FEATURE_EXTRACTORS = [feature_extraction.mean, feature_extraction.stdev, feature_extraction.correlate]


def pipeline(experiment_names):
    """Implements a segmentation-preprocessing-feature_extraction pipeline."""
    assert len(experiment_names) > 0

    segments = []

    for experiment_name in experiment_names:
        data_collection = DataCollection(experiment_name)
        data_collection.load()
        segments.extend(data_collection.segment())

    features = feature_extraction.extract_features(segments, FEATURE_EXTRACTORS)
    labels = np.array([segment.label for segment in segments])

    return features, labels


def train_pipeline(experiment_names, model_filename):
    """Trains a model on all the data provided and dumps it to a file."""
    features, labels = pipeline(experiment_names)

    model = train.train(X=features, y=labels)

    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)


def cross_validate_pipeline(experiment_names):
    """Trains and validates a model"""
    features, labels = pipeline(experiment_names)

    train.cross_validate(X=features, y=labels)


if __name__ == '__main__':
    pipeline_type = sys.argv[1]

    if pipeline_type == 'train':
        train_pipeline(experiment_names=sys.argv[3:], model_filename=sys.argv[2])
    elif pipeline_type == 'cross':
        cross_validate_pipeline(experiment_names=sys.argv[2:])
    else:
        print('Unrecognized pipeline: ' + pipeline_type)
