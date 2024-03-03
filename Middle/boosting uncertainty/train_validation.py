from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator < 1] = 1
    return np.mean(numerator / denominator)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y_pred - y_true) / np.sum(np.abs(y_true))



class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test
    gap : int, default=0
        Number of groups between train and test sets
    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                    'b', 'b', 'b', 'b', 'b',\
                    'c', 'c', 'c', 'c',\
                    'd', 'd', 'd',
                    'e', 'e', 'e'])
    >>> splitter = GroupTimeSeriesSplit(n_splits=3, max_train_size=2, gap=1)
    >>> for i, (train_idx, test_idx) in enumerate(
    ...     splitter.split(groups, groups=groups)):
    ...     print(f"Split: {i + 1}")
    ...     print(f"Train idx: {train_idx}, test idx: {test_idx}")
    ...     print(f"Train groups: {groups[train_idx]},
                    test groups: {groups[test_idx]}\n")
    Split: 1
    Train idx: [0 1 2 3 4 5], test idx: [11 12 13 14]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a'], test groups: ['c' 'c' 'c' 'c']

    Split: 2
    Train idx: [ 0  1  2  3  4  5  6  7  8  9 10], test idx: [15 16 17]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b'],
    test groups: ['d' 'd' 'd']

    Split: 3
    Train idx: [ 6  7  8  9 10 11 12 13 14], test idx: [18 19 20]
    Train groups: ['b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c'],
    test groups: ['e' 'e' 'e']
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        # Get unique groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Calculate number of groups for training and testing
        train_groups = self.n_splits * (self.max_train_size + self.gap)
        if self.test_size is None:
            test_groups = n_groups - train_groups
        else:
            test_groups = self.test_size

        # Validate parameters
        if train_groups >= n_groups:
            raise ValueError("Number of groups for training exceeds total number of groups.")
        if test_groups <= 0:
            raise ValueError("Test size is invalid.")

        # Generate indices for each split
        for i in range(self.n_splits):
            # Calculate starting index for training set
            train_start = i * (self.max_train_size + self.gap)
            train_end = train_start + self.max_train_size

            # Calculate starting index for test set
            test_start = train_end + self.gap
            test_end = test_start + test_groups

            # Get indices for training and test sets
            train_indices = np.where(np.isin(groups, unique_groups[train_start:train_end]))[0]
            test_indices = np.where(np.isin(groups, unique_groups[test_start:test_end]))[0]

            yield train_indices, test_indices


def best_model() -> Any:
    # YOU CODE IS HERE:
    # type your own sklearn model
    # with custom parameters
    pass
    #model = GradientBoostingRegressor(...)
    # ...
    #return model
