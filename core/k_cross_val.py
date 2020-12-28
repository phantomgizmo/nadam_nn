import random
import core.neural as neural
from typing import Tuple

XYWrapped = Tuple[list, list]

def create_folds(fold: int, wrapped_data: XYWrapped):
    """Create folds of data used."""
    X_data, y_data = wrapped_data
    fold_size = round(len(X_data) / fold)
    for i in range(fold):
        if i == fold-1:
            yield (X_data[i*fold_size:], y_data[i*fold_size:])
            break
        yield (X_data[i*fold_size:fold_size], y_data[i*fold_size:fold_size])

def flatten_list(data):
    """Data contains list of (X_data, y_data) which used for training.
    Return sum of X_data and y_data."""
    X_data = []
    y_data = []
    for X, y in data:
        X_data += X
        y_data += y

    return (X_data, y_data)

def k_cross_val(wrapped_data: XYWrapped, neural: neural, fold: int, epoch: int, min_error: float, random_state: int):
    X_data, y_data = wrapped_data
    #shuffle data
    random.Random(random_state).shuffle(X_data)
    random.Random(random_state).shuffle(y_data)
    #create folds of data
    folds_wrapped_data = list(create_folds(fold, (X_data, y_data)))
    #do k cross val
    for i in range(fold):
        left_folds_training_data = folds_wrapped_data[:i]
        right_folds_training_data = folds_wrapped_data[i+1:]
        training_data = left_folds_training_data + right_folds_training_data
        training_data = flatten_list(training_data)
        test_data = folds_wrapped_data[i]

        # neural.NAdam(training_data=)