from collections import Counter
from pathlib import Path

import fire
import numpy as np
import pandas as pd

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

    # balance 0 and 7
    assert counter[0] > counter[7]
    multiplier, remainder = divmod(counter[0] - counter[7], counter[7])
    indices = count_reversed[7] * multiplier + count_reversed[7][:remainder]
    assert matrix[indices].shape[0] == counter[0] - counter[7]
    matrix = np.vstack([matrix, matrix[indices]])

    # balance 1 and 6
    assert counter[1] > counter[6]
    multiplier, remainder = divmod(counter[1] - counter[6], counter[6])
    indices = count_reversed[6] * multiplier + count_reversed[6][:remainder]
    assert matrix[indices].shape[0] == counter[1] - counter[6]
    matrix = np.vstack([matrix, matrix[indices]])

    # balance 2 and 5
    assert counter[2] > counter[5]
    multiplier, remainder = divmod(counter[2] - counter[5], counter[5])
    indices = count_reversed[5] * multiplier + count_reversed[5][:remainder]
    assert matrix[indices].shape[0] == counter[2] - counter[5]
    matrix = np.vstack([matrix, matrix[indices]])

    # balance 3 and 4
    assert counter[3] > counter[4]
    multiplier, remainder = divmod(counter[3] - counter[4], counter[4])
    indices = count_reversed[4] * multiplier + count_reversed[4][:remainder]
    assert matrix[indices].shape[0] == counter[3] - counter[4]
    matrix = np.vstack([matrix, matrix[indices]])

    # save resulting matrix
    np.save('events_oversampled', matrix)


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


if __name__ == '__main__':
    fire.Fire(oversample)
