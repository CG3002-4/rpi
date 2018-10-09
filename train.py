import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle


NUM_ESTIMATORS = 39


def train_internal(X, y):
    """Expects input data to be shuffled."""
    max_features = (np.sqrt(len(X.columns)) + 1) / len(X.columns)

    clf = ExtraTreesClassifier(random_state=0, max_features=max_features,
                               n_estimators=NUM_ESTIMATORS, max_depth=None,
                               min_samples_split=2, bootstrap=False)
    clf.fit(X, y)

    return clf


def cross_validate(X, y):
    """Uses 10-fold validation to train and test a model."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    skf.get_n_splits(X)
    confusion_matrices = []
    accuracy = []
    feature_impt_list = []

    for train_index, test_index in skf.split(X, y):
        train_X, test_X = X.iloc[train_index], X.iloc[test_index]
        train_y, test_y = y[train_index], y[test_index]

        clf = train_internal(train_X, train_y)
        predictions = clf.predict(test_X)
        accuracy.append(accuracy_score(test_y, predictions))
        confusion_matrices.append(confusion_matrix(test_y, predictions))
        feature_impt_list.append(clf.feature_importances_)

    # Display various stats
    np.set_printoptions(suppress=True)

    print("Accuracy: " + str(np.mean(accuracy)))

    confusion_avg = np.mean(confusion_matrices, axis=0)
    print("Confusion matrix:")
    print(confusion_avg)

    # View a list of the features and their importance scores
    feature_impt_list = np.mean(feature_impt_list, axis=0)
    feature_importance = list(zip(X, feature_impt_list))
    feature_importance.sort(key=lambda x: x[1])
    print("Feature Importance:")
    for feature, importance in feature_importance:
        print(feature + ": " + str(importance))

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
    X, y = shuffle(X, y)

    return train_internal(X, y)
