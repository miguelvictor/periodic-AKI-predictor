from pathlib import Path
from typing import Dict, List, Tuple

import fire
import numpy as np
import pandas as pd
import random

TIMESTEPS = 8
N_FEATURES = 16


def undersample(name: str = 'events_complete.csv', dataset_dir: str = 'dataset'):
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
    print(f'Padding samples up to {TIMESTEPS} days.')
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
    print(f'Splitting samples for training, validation, and testing sets.')
    train_indices = []
    val_indices = []
    test_indices = []
    for los in range(2, TIMESTEPS + 1):
        for i_train, i_val, i_test in split_los_samples(stats, index_mapping, los):
            train_indices.extend(i_train)
            val_indices.extend(i_val)
            test_indices.extend(i_test)

    # shuffle indices
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    # save resulting matrices
    np.save(dataset_dir / 'finetuning_training', matrix[train_indices])
    np.save(dataset_dir / 'finetuning_validation', matrix[val_indices])
    np.save(dataset_dir / 'finetuning_testing', matrix[test_indices])


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
    for i in range(int(los / 2)):
        set1 = stats[(stats['los'] == los) & (stats['aki'] == i)]
        set2 = stats[(stats['los'] == los) & (stats['aki'] == los - 1 - i)]

        # split samples for training/validation/testing sets
        set1_train, set1_val, set1_test = split_indices(set1.index)
        set2_train, set2_val, set2_test = split_indices(set2.index)

        yield (
            [mapping[index] for index in set1_train + set2_train],
            [mapping[index] for index in set1_val + set2_val],
            [mapping[index] for index in set1_test + set2_test],
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
    # get the total count of aki for each ICU stays
    aki_count = pd.pivot_table(
        df,
        index='stay_id',
        values='aki',
        aggfunc=np.sum,
    )

    # get the length of stay of each ICU stays
    los = pd.pivot_table(
        df,
        index='stay_id',
        values='aki',
        aggfunc=len,
    )

    aki_los = pd.concat([aki_count, los], axis=1)
    aki_los.columns = ['aki', 'los']
    aki_los['aki'] = aki_los['aki'].astype('int')

    return aki_los


if __name__ == '__main__':
    fire.Fire(undersample)
