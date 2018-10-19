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
                      feature_extraction.var,
                      feature_extraction.correlate,
                      feature_extraction.max,
                      feature_extraction.min]


def feature_extraction_pipeline(exp_names):
    """Implements a segmentation-preprocessing-feature_extraction pipeline."""
    segments = []

    for exp_name in exp_names:
        exp_dir = os.path.join(DATA_FOLDER, exp_name)
        data_collection = DataCollection(exp_dir)
        data_collection.load()
        segments.extend(data_collection.segment())

    # preprocessed_segments = preprocess.preprocess_segments(
    #     segments, NOISE_FILTERS)
    features = feature_extraction.extract_features(
        segments, FEATURE_EXTRACTORS)
    labels = np.array([segment.label for segment in segments])

    # neutral_idxs = labels != 1
    # return features[neutral_idxs], labels[neutral_idxs]
    return features, labels


def save_features_and_labels(features, labels, filename):
    # Create new DataFrame with both features and labels
    data = features.copy()
    data['label'] = pd.Series(labels, index=data.index)
    data.to_csv(filename + '.csv', index=False)


def load_features_and_labels(filename):
    data = pd.read_csv(filename + '.csv', dtype=float)
    return data.loc[:, data.columns[:-1]], data['label']


def train_pipeline(X, y, model_filename):
    """Trains a model on all the data provided and dumps it to a file."""
    model = train.train(X, y)

    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)


def cross_validate_pipeline(X, y):
    """Trains and validates a model"""
    train.cross_validate(X, y)


def get_all_experiments():
    return [x[1] for x in os.walk(DATA_FOLDER)][0]


def exactly_one_true(*bools):
    return bools.count(True) == 1


def get_features_and_labels_from_args(args):
    assert exactly_one_true(
        args.load_file is not None,
        len(args.exp_names) > 0,
        args.all_exp
    ), 'Must specify exactly one of loading from file, loading from particular experiments and loading from all experiments'

    if args.load_file is not None:
        return load_features_and_labels(args.load_file)
    elif args.all_exp:
        return feature_extraction_pipeline(get_all_experiments())
    else:
        return feature_extraction_pipeline(args.exp_names)


def create_save_features_parser(subparsers):
    parser = subparsers.add_parser(
        'save-features',
        description='Extract features from experiment(s) and save them to a csv file. Load data from EITHER --exp-names OR all-exp. Do not add both as arguments.'
    )
    parser.add_argument(
        '--data-file', help='Filename of csv to which features will be written')
    parser.add_argument('--exp-names', nargs='+',
                        default=[], help='Names of experiment(s) to extract from')
    parser.add_argument('--all-exp', action='store_true', default=False,
                        help='Extract from all experiments')

    def save_features_with_args(args):
        assert exactly_one_true(
            args.all_exp,
            len(args.exp_names) > 0
        ), 'Must specify exactly one of particular experiment names and all experiments'

        if args.all_exp:
            features, labels = feature_extraction_pipeline(
                get_all_experiments())
        else:
            features, labels = feature_extraction_pipeline(
                args.exp_names)
        save_features_and_labels(features, labels, args.data_file)

    parser.set_defaults(func=save_features_with_args)


def add_loading_arguments(parser):
    parser.add_argument(
        '--load-file', help='Name of csv file containing features to be loaded')
    parser.add_argument('--exp-names', nargs='+',
                        default=[], help='Names of experiment(s) to train on')
    parser.add_argument('--all-exp', action='store_true',
                        help='Train on all experiments')


def create_train_parser(subparsers):
    parser = subparsers.add_parser('train',
                                   description='Train model from EITHER --load-file OR --exp-names OR --all-exp. Do not add more than one as arguments.')
    parser.add_argument(
        '--model-file', help='Name of pickle file in which to dump trained model.')
    add_loading_arguments(parser)

    def train_with_args(args):
        features, labels = get_features_and_labels_from_args(args)
        train_pipeline(X=features, y=labels,
                       model_filename=args.model_file + '.pb')

    parser.set_defaults(func=train_with_args)


def create_cross_validation_parser(subparsers):
    parser = subparsers.add_parser('cross',
                                   description='Cross-validate model from EITHER --load-file OR --exp-names OR --all-exp. Do not add more than one as arguments.')
    add_loading_arguments(parser)

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
