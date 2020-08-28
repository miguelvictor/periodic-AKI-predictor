from collections import Counter

import numpy as np


def convert_preds(x, y, y_hat):
    # get the indices of the last day of a specific ICU stay
    indices = np.any(x, axis=-1).sum(axis=-1) - 1

    # create a new y and y_hat only considering the values
    # of the last day predictions
    _y = []
    _y_hat = []

    # TODO: somehow refactor this so that it doesn't use a for-loop
    for i in range(y.shape[0]):
        last_day_index = indices[i]
        _y.append(y[i, last_day_index, 0])
        _y_hat.append(y_hat[i, last_day_index, 0])

    return (
        np.array(_y).astype(np.float32),
        np.array(_y_hat).astype(np.float32),
    )


def early_prediction_score(y_true, y_pred):
    #y_true = y_true.reshape(y_true.shape[0], -1)
    #y_pred = y_pred.reshape(y_pred.shape[0], -1)
    assert y_true.shape == y_pred.shape

    n_correct = 0
    stats = Counter()

    for i in range(y_true.shape[0]):
        # first aki positive index
        true_i, = np.where(y_true[i] == 1)
        pred_i, = np.where(y_pred[i] == 1)

        # y_true has aki label, but y_pred doesn't have
        # and the other way around
        if (
            len(true_i) != 0 and len(pred_i) == 0 or
            len(pred_i) != 0 and len(true_i) == 0
        ):
            continue

        # no aki labels
        if len(true_i) == 0 and len(pred_i) == 0:
            n_correct += 1
            continue

        # reassign values to first aki-day index
        true_i, pred_i = true_i[0], pred_i[0]

        # aki was detected early
        if pred_i <= true_i:
            n_correct += 1
            stats[true_i - pred_i] += 1

    return n_correct / y_true.shape[0], stats
