"""Implements the training and validation pipelines.

Run `python3 pipeline.py -h` to see help
Run `python3 pipeline.py <pipeline_type> -h` to see help for that pipeline
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from data_collection import DataCollection
import preprocess
import feature_extraction
import train


DATA_FOLDER = 'data'
NOISE_FILTERS = [preprocess.medfilt, preprocess.butter_noise]
FEATURE_EXTRACTORS = [feature_extraction.mean,
                      feature_extraction.stdev,
                      feature_extraction.correlate]


def feature_extraction_pipeline(experiment_names):
    """Implements a segmentation-preprocessing-feature_extraction pipeline."""
    segments = []

    for experiment_name in experiment_names:
        experiment_dir = os.path.join(DATA_FOLDER, experiment_name)
        data_collection = DataCollection(experiment_dir)
        data_collection.load()
        segments.extend(data_collection.segment())

    preprocessed_segments = preprocess.preprocess_segments(
        segments, NOISE_FILTERS)
    features = feature_extraction.extract_features(
        preprocessed_segments, FEATURE_EXTRACTORS)
    labels = np.array([segment.label for segment in segments])

    return features, labels


def save_features_and_labels(features, labels, filename):
    # Create new DataFrame with both features and labels
    data = features.copy()
    data['label'] = pd.Series(labels, index=data.index)
    data.to_csv(filename)

    
def load_features_and_labels(filename):
    data = pd.DataFrame.from_csv(filename)
    return data.loc[:, data.columns[:-1]], data['label']


def train_pipeline(X, y, model_filename):
    """Trains a model on all the data provided and dumps it to a file."""
    model = train.train(X, y)

    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)


def cross_validate_pipeline(X, y):
    """Trains and validates a model"""
    train.cross_validate(X, y)


def get_features_and_labels_from_args(args):
    # Note the use of the xor operator here
    assert (args.load_file is None) ^ len(args.experiment_names) == 0, 'Must specify exactly one of loading from file and loading from experiments'
    if args.load_file is not None:
        return load_features_and_labels(args.load)
    else:
        return feature_extraction_pipeline(args.experiment_names)


def create_save_features_parser(subparsers):
    parser = subparsers.add_parser(
        'save-features',
        description='Extract features from experiments and save them to a file'
    )
    parser.add_argument('--data-file', help='File to which features will be written')
    parser.add_argument('--experiment-names', nargs='+', default=[], help='Names of experiments to extract from')

    def save_features_with_args(args):
        features, labels = feature_extraction_pipeline(args.experiment_names)
        save_features_and_labels(features, labels, args.data_file)

    parser.set_defaults(func=save_features_with_args)


def create_train_parser(subparsers):
    parser = subparsers.add_parser('train')
    parser.add_argument('--load-file', help='File containing features to be loaded')
    parser.add_argument('--model-file', help='File in which to dump trained model')
    parser.add_argument('--experiment-names', nargs='+', default=[], help='Names of experiments to train on')

    def train_with_args(args):
        features, labels = get_features_and_labels_from_args(args)
        train_pipeline(X=features, y=labels, model_filename=args.model_file)

    parser.set_defaults(func=train_with_args)


def create_cross_validation_parser(subparsers):
    parser = subparsers.add_parser('cross')
    parser.add_argument('--load-file', help='File containing features to be loaded')
    parser.add_argument('--experiment-names', nargs='+', default=[], help='Names of experiments to cross-validate on')

    def cross_validate_with_args(args):
        features, labels = get_features_and_labels_from_args(args)
        cross_validate_pipeline(X=features, y=labels)

    parser.set_defaults(func=cross_validate_with_args)


def create_parser():
    parser = argparse.ArgumentParser(
        description='Various pipelines for machine learning')
    subparsers = parser.add_subparsers(title='available pipelines')
    create_save_features_parser(subparsers)
    create_train_parser(subparsers)
    create_cross_validation_parser(subparsers)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args(sys.argv[1:])

    if len(args.__dict__) <= 1:
        # No subcommand specified
        parser.print_help()
        parser.exit()

    args.func(args)
