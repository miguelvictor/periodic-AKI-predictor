from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import numpy as np
import pandas as pd
import random

TIMESTEPS = 8
N_FEATURES = 16


def oversample(name: str = 'events_complete.csv', dataset_dir: str = 'dataset'):
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f'{name} does not exist. Run `extract-dataset.py` first')

    # load input csv
    df = pd.read_csv(path)
    df.columns = map(str.lower, df.columns)

    # get los and corresponding aki label for each ICU stay
    stats = get_statistics(df)
    index_mapping = dict(zip(stats.index, range(len(stats))))

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

    # balance 2, 3, 4, 5, 6, 7, 8 los
    train_indices = []
    val_indices = []
    test_indices = []
    for los in range(2, TIMESTEPS + 1):
        i_train, i_val, i_test = split_los_samples(stats, index_mapping, los)
        train_indices.extend(i_train)
        val_indices.extend(i_val)
        test_indices.extend(i_test)

    # shuffle indices
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    # save resulting matrices
    np.save(dataset_dir / 'matrix_training', matrix[train_indices])
    np.save(dataset_dir / 'matrix_validation', matrix[val_indices])
    np.save(dataset_dir / 'matrix_testing', matrix[test_indices])


def padding(group) -> pd.DataFrame:
    '''
    Adds padding to a group so that the resulting dataframe will have
    a number of rows equal to the timesteps constant. Also, stay_id
    column is preserved on the padding rows.
    '''
    n_rows, n_cols = group.shape
    n_rows = TIMESTEPS - n_rows
    padding = np.zeros((n_rows, n_cols))
    padding = pd.DataFrame(padding, columns=group.columns)
    padding['stay_id'] = group['stay_id'].iloc[0]
    return pd.concat([group, padding], axis=0)


def split_los_samples(stats: pd.DataFrame, mapping: Dict[int, int], los: int) -> Tuple[List[int], List[int], List[int]]:
    '''
    Takes in `stats` (a dataframe that contains the aki label and los 
    for each ICU stay) and balances out the samples so that the number of 
    positive and negative samples is equal for the given los.
    '''
    pos = stats[(stats['los'] == los) & (stats['aki'] == 1)]
    neg = stats[(stats['los'] == los) & (stats['aki'] == 0)]

    p_train, p_val, p_test = split_indices(pos.index)
    n_train, n_val, n_test = split_indices(neg.index)

    # medical datasets tend to have big dataset imbalance
    # so we duplicate positive samples to match its count with the negative samples
    p_train = duplicate(p_train, t=len(n_train))
    p_val = duplicate(p_val, t=len(n_val))
    p_test = duplicate(p_test, t=len(n_test))

    return (
        [mapping[index] for index in p_train + n_train],
        [mapping[index] for index in p_val + n_val],
        [mapping[index] for index in p_test + n_test],
    )


def split_indices(indices: pd.Int64Index) -> Tuple[List[int], List[int], List[int]]:
    '''
    Splits the given sample indices into three partitions 
    (for training, validation, and testing sets).
    If the sample count is not enough, testing and validation sets are given priority.
    '''
    length = len(indices)
    training = []
    validation = []
    testing = []

    # use normal 80%, 10%, 10% split
    if length >= 10:
        training_end_index = int(length * .8)
        validation_end_index = int(length * .9)
        training.extend(indices[:training_end_index])
        validation.extend(indices[training_end_index:validation_end_index])
        testing.extend(indices[validation_end_index:])
    elif length == 1:
        testing.append(indices[0])
    elif length == 2:
        testing.append(indices[0])
        validation.append(indices[1])
    else:
        testing.append(indices[0])
        validation.append(indices[1])
        training.extend(indices[2:])

    return training, validation, testing


def get_statistics(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Computes the relevant statistics of each ICU stay in the dataframe.
    It returns a dataframe with an aki label and los for each ICU stay.
    '''
    # get the last day predictions of each ICU stays
    last_day_preds = pd.pivot_table(
        df,
        index='stay_id',
        values='aki',
        aggfunc=lambda x: x.iloc[-1]
    )

    # get the length of stay of each ICU stays
    los = pd.pivot_table(
        df,
        index='stay_id',
        values='aki',
        aggfunc=len
    )

    aki_los = pd.concat([last_day_preds, los], axis=1)
    aki_los.columns = ['aki', 'los']
    aki_los['aki'] = aki_los['aki'].astype('int')

    return aki_los


def duplicate(indices: List[int], t: int = 0) -> List[int]:
    '''
    Duplicates the contents of indices so that its length
    will be equal to `t`. However, if indices is empty, then
    an empty list is also returned.
    '''
    f = len(indices)
    if f == 0:
        return []

    assert t > f
    quotient, remainder = divmod(t, f)
    indices = indices * quotient + indices[:remainder]

    assert len(indices) == t
    return indices


if __name__ == '__main__':
    fire.Fire(oversample)
