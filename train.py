import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


NUM_ESTIMATORS = 39
MEANING_OF_LIFE = 42
SEEDS = [85, 37, 88, 33, 58, 70, 12, 62, 6, 35]
# SEEDS = [67, 40, 30, 71, 95, 44, 66, 94, 79, 43, 86, 51, 31, 76,
#          88, 48, 25, 7, 59, 14, 82, 29, 13, 68, 26, 1, 52, 0, 80, 87]


def train_internal(X, y, classifier, random_state, bootstrap):
    """Expects input data to be shuffled.

    If no value given for random_state, then will be chosen by code.
    """
    MAX_FEATURES = (np.sqrt(X.shape[1]) + 1) / X.shape[1]

    clf = classifier(random_state=random_state, max_features=MAX_FEATURES,
                     n_estimators=NUM_ESTIMATORS, max_depth=None,
                     min_samples_split=2, bootstrap=bootstrap)
    clf.fit(X, y)

    return clf


def cross_validate(X, y):
    """Uses 10-fold validation to train and test a model."""

    # Display various stats
    np.set_printoptions(suppress=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf.get_n_splits(X)

    classifiers = [#(RandomForestClassifier, True),
                   (ExtraTreesClassifier, False)]

    for classifier, bootstrap in classifiers:
        print(classifier)
        print(SEEDS)
        confusion_matrices = []
        accuracy = []
        feature_impt_list = []

        for seed in SEEDS:
            for train_index, test_index in skf.split(X, y):
                train_X, test_X = X[train_index], X[test_index]
                train_y, test_y = y[train_index], y[test_index]

                # std_scale = StandardScaler().fit(train_X)
                # train_X = std_scale.transform(train_X)
                # test_X = std_scale.transform(test_X)

                # clf = OneVsRestClassifier(LinearSVC(random_state=seed))
                # clf.fit(train_X, train_y)

                clf = train_internal(
                    train_X, train_y, classifier, random_state=seed, bootstrap=bootstrap)
                predictions = clf.predict(test_X)
                accuracy.append(accuracy_score(test_y, predictions))
                confusion_matrices.append(
                    confusion_matrix(test_y, predictions))
                feature_impt_list.append(clf.feature_importances_)

        print("Accuracy: " + str(np.mean(accuracy)))

        confusion_avg = np.mean(confusion_matrices, axis=0)
        print("Confusion matrix:")
        print(confusion_avg)

        # View a list of the features and their importance scores
        feature_impt_list = np.mean(feature_impt_list, axis=0)
        feature_importance = list(zip(range(X.shape[1]), feature_impt_list))
        feature_importance.sort(key=lambda x: x[1])
        print("Feature Importance:")
        for feature, importance in feature_importance:
            print(str(feature) + ": " + str(importance))

        import matplotlib.pyplot as plt

        confusion_norm = confusion_avg / np.sum(confusion_avg, axis=1)

        plt.figure()
        plt.imshow(confusion_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Normalized confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(confusion_norm))
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)

        fmt = '.2f'
        thresh = confusion_norm.max() / 2.
        for i, j in itertools.product(range(confusion_norm.shape[0]), range(confusion_norm.shape[1])):
            plt.text(j, i, format(confusion_norm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_norm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


def train(X, y):
    # Shuffle the X and y values in unison
    X, y = shuffle(X, y, random_state=MEANING_OF_LIFE)

    return train_internal(X, y, ExtraTreesClassifier, random_state=MEANING_OF_LIFE, bootstrap=False)


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('feature.csv')
    features, labels = data[:, :-1], data[:, -1]

    cross_validate(features, labels)
