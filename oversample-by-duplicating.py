from collections import Counter
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import random

TIMESTEPS = 8
N_FEATURES = 16


def oversample(name: str = 'events_complete.csv', dataset_dir: str = 'dataset'):
    path = Path(dataset_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f'{name} does not exist. Run `extract-dataset.py` first')

    # load input csv
    df = pd.read_csv(path)
    df.columns = map(str.lower, df.columns)

    # define columns to keep and to discard
    to_discard = ['stay_day', 'stay_id', 'subject_id', 'hadm_id']
    columns = [c for c in df.columns if c not in to_discard]
    assert len(columns) == N_FEATURES + 1  # plus AKI label

    # pad dataframe so that each ICU stay will be 8 days
    # also add sanity checks
    n_stays_before = len(pd.unique(df['stay_id']))
    df = df.groupby('stay_id').apply(padding).reset_index(drop=True)
    n_stays_after = len(pd.unique(df['stay_id']))
    n_stays, remainder = divmod(df.shape[0], TIMESTEPS)
    assert n_stays_before == n_stays_after
    assert n_stays == n_stays_after
    assert remainder == 0

    # transform pandas dataframe into numpy 2d array
    # and remove unnecessary columns
    matrix = df[columns].values.reshape(-1, TIMESTEPS, N_FEATURES + 1)

    # compute each ICU stay's positive count
    counter, count_reversed = get_statistics(matrix)
    _, max_len = counter.most_common(1)[0]

    # balance 1, 2, 3, 4, 5, 6, 7
    for i in range(1, TIMESTEPS):
        indices = duplicate(count_reversed[i], f=counter[i], t=max_len)
        count_reversed[i].extend(indices)

    # after balancing, all labels should have the same length
    # shuffle the indices in preparation for splits
    for k in count_reversed.keys():
        assert len(count_reversed[k]) == max_len
        random.shuffle(count_reversed[k])

    train_indices = []
    validation_indices = []
    test_indices = []
    train_end = int(max_len * .8)
    validation_end = int(max_len * .9)

    for i in range(TIMESTEPS):
        indices = count_reversed[i]
        train_indices.extend(indices[:train_end])
        validation_indices.extend(indices[train_end:validation_end])
        test_indices.extend(indices[validation_end:])

    # save resulting matrices
    np.save(dataset_dir / 'matrix_training', matrix[train_indices])
    np.save(dataset_dir / 'matrix_validation', matrix[validation_indices])
    np.save(dataset_dir / 'matrix_testing', matrix[test_indices])


def padding(group):
    n_rows, n_cols = group.shape
    n_rows = TIMESTEPS - n_rows
    padding = np.zeros((n_rows, n_cols))
    padding = pd.DataFrame(padding, columns=group.columns)
    padding['stay_id'] = group['stay_id'].iloc[0]
    return pd.concat([group, padding], axis=0)


def get_statistics(matrix):
    # compute the number of positive count for each ICU stay
    # resulting shape: [n_samples]
    p_count = np.sum(matrix[:, :, -1], axis=1).astype('int')
    counter = Counter(p_count)

    # store indices of ICU stays with a certain amount of positive count
    # to be used for duplication purposes
    p_count_reversed = {}
    for i, count in enumerate(p_count):
        if count not in p_count_reversed:
            p_count_reversed[count] = [i]
        else:
            p_count_reversed[count].append(i)

    return counter, p_count_reversed


def duplicate(indices, f=0, t=0):
    assert t > f
    quotient, remainder = divmod(t - f, f)
    indices = indices * quotient + indices[:remainder]
    assert len(indices) == t - f
    return indices


if __name__ == '__main__':
    fire.Fire(oversample)
