from pathlib import Path

import fire
import logging
import numpy as np
import pandas as pd
import re

DB_PATH = Path('databases')
MIMIC4_PATH = DB_PATH / 'mimic4'
N_FEATURES = 16
TIMESTEPS = 8

logging.basicConfig(
    filename='prepare-pretraining-data.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger('default')

LABEVENTS_FEATURES = {
    'bicarbonate': [50882],  # mEq/L
    'chloride': [50902],
    'creatinine': [50912],
    'glucose': [50931],
    'magnesium': [50960],
    'potassium': [50822, 50971],  # mEq/L
    'sodium': [50824, 50983],  # mEq/L == mmol/L
    'bun': [51006],
    'hemoglobin': [51222],
    'platelets': [51265],
    'wbcs': [51300, 51301],
}
CHARTEVENTS_FEATURES = {
    'height': [
        226707,  # inches
        226730,  # cm
        1394,  # inches
    ],
    'weight': [
        763,  # kg
        224639,  # kg
    ]
}


def partition_rows(input_path, output_path):
    '''
    Reads in the combined chartevents and labevents csv file (filtered_events.csv)
    and aggregates the features values with respect to each ICU day of the different 
    patients (feature values as the columns, ICU day as the rows).

    Parameters:
    input_path: the path of the input csv to be processed (e.g., filtered_events.csv)
    output_path: the path as to where the output of this step should be dumped
    '''
    logger.info('`partition_rows` has started')
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # extract the day of the event
    df['chartday'] = df['charttime'].astype(
        'str').str.split(' ').apply(lambda x: x[0])

    # group day into a specific ICU stay
    df['stay_day'] = df['stay_id'].astype('str') + '_' + df['chartday']

    # add feature label column
    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}
    features_reversed = {v2: k for k, v1 in features.items() for v2 in v1}
    df['feature'] = df['itemid'].apply(lambda x: features_reversed[x])

    # save mapping of icu stay ID to patient ID
    icu_subject_mapping = dict(zip(df['stay_day'], df['subject_id']))

    # convert height (inches to cm)
    mask = (df['itemid'] == 226707) | (df['itemid'] == 1394)
    df.loc[mask, 'valuenum'] *= 2.54

    # average all feature values each day
    df = pd.pivot_table(
        df,
        index='stay_day',
        columns='feature',
        values='valuenum',
        fill_value=np.nan,
        aggfunc=np.nanmean,
        dropna=False,
    )

    # insert back information related to the patient (for persistence)
    df['stay_day'] = df.index
    df['stay_id'] = df['stay_day'].str.split(
        '_').apply(lambda x: x[0]).astype('int')
    df['subject_id'] = df['stay_day'].apply(lambda x: icu_subject_mapping[x])

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`partition_rows` has ended')


def impute_holes(input_path, output_path):
    '''
    Fills in NaN values using forward/backward imputation.
    Entries that doesn't meet some imposed criteria will be dropped.

    Parameters:
    input_path: the path of the input csv to be processed (e.g., events_partitioned.csv)
    output_path: the path as to where the output of this step should be dumped
    '''
    logger.info('`impute_holes` has started')
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # collect all feature keys
    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}.keys()

    # fill NaN values with the average feature value (only for the current ICU stay)
    # ICU stays with NaN average values are dropped
    stay_ids = pd.unique(df['stay_id'])
    logger.info(f'Total ICU stays: {len(stay_ids)}')

    for stay_id in stay_ids:
        # get mask for the current icu stay
        stay_id_mask = df['stay_id'] == stay_id

        # there are ICU stays that even though its los >= 3
        # the actual measurements done in labevents or chartevents are fewer than that
        # so we drop them here
        if df[stay_id_mask].shape[0] < 3:
            logger.warning(f'ICU stay id={stay_id} has los<3 (dropped)')
            df = df[~stay_id_mask]
            continue

        # drop ICU stays with no creatinine levels
        # after the first 48 hours
        if not np.isfinite(df[stay_id_mask]['creatinine'].values[2:]).any():
            logger.warning(f'ICU stay id={stay_id} creatinine levels'
                           + ' are all NaN after 48 hours (dropped)')
            df = df[~stay_id_mask]
            continue

        # drop ICU stays with no creatinine levels
        # at the third day
        nan_index = get_nan_index(df[stay_id_mask]['creatinine'])
        if nan_index == 2:
            logger.warning(f'ICU stay id={stay_id} creatinine level'
                           + ' at 3rd day is not available (dropped)')
            df = df[~stay_id_mask]
            continue

        # drop ICU stay days (and onwards) with no creatinine levels defined
        if nan_index != -1:
            logger.warning(f'ICU stay id={stay_id} creatinine level'
                           + f' at {nan_index}th day is not available (dropped)')
            nan_indices = df[stay_id_mask].index[nan_index:]
            df = df.drop(nan_indices)

        # fill feature missing values with the mean value
        # of the ICU stay, dropping ICU stays with missing values
        df = fill_nas_or_drop(df, stay_id, features)

    # stay IDs whose weight and height values are missing
    # we use the global mean to impute these
    df['height'].fillna(df['height'].mean(), inplace=True)
    df['weight'].fillna(df['weight'].mean(), inplace=True)

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`impute_holes` has ended')


def fill_nas_or_drop(df, stay_id, features):
    '''
    A helper function to the impute_holes function. This does the actual
    forward/backward imputation which fills the NaN values. Also, entries
    without valid values for the whole ICU stay span will be dropped.

    Parameters:
    df: The input dataframe to be processed.
    stay_id: The ID of the ICU stay of a certain patient to be processed.
    features: The features used in this work (defined at the top as a constant).
    '''
    # get mask for the current icu stay
    stay_id_mask = df['stay_id'] == stay_id

    for feature in features:
        # drop ICU stays with features that doesn't contain any
        # finite values (e.g., all values are NaN)
        entity_features = df.loc[stay_id_mask, feature]
        if not np.isfinite(entity_features).any():
            # missing height and weight values will be imputed with the global mean
            if feature == 'weight' or feature == 'height':
                continue

            # other feature values are important
            logger.warning(f'ICU stay id={stay_id} feature={feature}'
                           + ' does not contain valid values (dropped)')
            return df[~stay_id_mask]

    # we impute feature values using forward/backward fills
    df.loc[stay_id_mask] = df[stay_id_mask].ffill().bfill()

    return df


def add_patient_info(input_path, output_path):
    '''
    Adds the patient information (static) to each of the entries.

    Parameters:
    input_path: the path of the input csv to be processed (e.g., events_imputed.csv)
    output_path: the path as to where the output of this step should be dumped
    '''
    logger.info('`add_patient_info` has started')

    admissions_path = MIMIC4_PATH / 'filtered_admissions.csv'
    admissions = pd.read_csv(admissions_path)
    admissions.columns = map(str.lower, admissions.columns)

    icustays_path = MIMIC4_PATH / 'filtered_icustays.csv'
    icustays = pd.read_csv(icustays_path)
    icustays.columns = map(str.lower, icustays.columns)

    patients_path = MIMIC4_PATH / 'filtered_patients.csv'
    patients = pd.read_csv(patients_path)
    patients.columns = map(str.lower, patients.columns)

    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    stay_ids = pd.unique(df['stay_id'])
    logger.info(f'Total ICU stays: {len(stay_ids)}')

    # get auxiliary features
    hadm_id_mapping = dict(zip(icustays['stay_id'], icustays['hadm_id']))
    ethnicity_mapping = dict(
        zip(admissions['hadm_id'], admissions['ethnicity']))
    gender_mapping = dict(zip(patients['subject_id'], patients['gender']))
    age_mapping = dict(zip(patients['subject_id'], patients['anchor_age']))

    # retrieve admission ID from stay_day
    df['stay_id'] = df['stay_day'].str.split('_').apply(lambda x: x[0])
    df['stay_id'] = df['stay_id'].astype('int')
    df['hadm_id'] = df['stay_id'].apply(lambda x: hadm_id_mapping[x])

    # compute patient's age
    df['age'] = df['subject_id'].apply(lambda x: age_mapping[x])

    # add patient's gender
    df['gender'] = df['subject_id'].apply(lambda x: gender_mapping[x])
    df['gender'] = (df['gender'] == 'M').astype('int')

    # add patient's ethnicity (black or not)
    df['ethnicity'] = df['hadm_id'].apply(lambda x: ethnicity_mapping[x])
    df['black'] = df['ethnicity'].str.contains(
        r'.*black.*', flags=re.IGNORECASE).astype('int')

    # drop unneeded columns
    del df['ethnicity']

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`add_patient_info` has ended')


def get_nan_index(series):
    '''
    Returns the index of the first nan value within the series (-1 if there's
    no nan values in the series). This only considers the nan values from the 3rd element
    onwards since the first and second element are already checked and assumed to be non-nan.
    '''
    result = ~np.isfinite(series)
    for i, x in enumerate(result[2:]):
        if x:
            return i + 2

    return -1


def transform_outliers(input_path, output_path):
    '''
    Detects the presence of outliers and replaces their values 
    with the lower/upper bound.

    Parameters:
    input_path: the path of the input csv to be processed (e.g., events_with_labels.csv)
    output_path: the path as to where the output of this step should be dumped
    '''
    logger.info('`transform_outliers` has started')
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}
    for feature in features.keys():
        # there are some bizarre values (e.g., negative person weights)
        # most likely due to typos, so we correct them here
        upper_bound = df[feature].quantile(.99)
        lower_bound = df[feature].quantile(.01)
        logger.info(f'Feature={feature} upper bound={upper_bound}')
        logger.info(f'Feature={feature} lower bound={lower_bound}')

        upper_mask = df[feature] > upper_bound
        lower_mask = df[feature] < lower_bound
        upper_ids = pd.unique(df.loc[upper_mask, 'stay_id'])
        lower_ids = pd.unique(df.loc[lower_mask, 'stay_id'])

        if len(upper_ids) > 0:
            # rescale values to the upper bound
            logger.info(f'Feature={feature}, {upper_ids} contains +outliers')
            df.loc[upper_mask, feature] = upper_bound

        if len(lower_ids) > 0:
            # rescale values to the lower bound
            logger.info(f'Feature={feature}, {lower_ids} contains -outliers')
            df.loc[lower_mask, feature] = lower_bound

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`transform_outliers` has ended')


def transform_to_tensor(input_path, output_path):
    logger.info('`transform_outliers` has started')
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # define columns to keep and to discard
    to_discard = ['stay_day', 'stay_id', 'subject_id', 'hadm_id']
    columns = [c for c in df.columns if c not in to_discard]
    assert len(columns) == N_FEATURES

    # add padding to ICU stays whose los < timesteps
    n_stays_before = len(pd.unique(df['stay_id']))
    df = df.groupby('stay_id').apply(padding).reset_index(drop=True)
    n_stays_after = len(pd.unique(df['stay_id']))
    n_stays, remainder = divmod(df.shape[0], TIMESTEPS)
    assert n_stays_before == n_stays_after
    assert n_stays == n_stays_after
    assert remainder == 0

    # reshape tensor into a 3D tensor with shape [_, timesteps, n_features]
    matrix = df[columns].values.reshape(-1, TIMESTEPS, N_FEATURES)
    np.save(output_path, matrix)
    logger.info('`transform_outliers` has ended')


def padding(group) -> pd.DataFrame:
    '''
    Adds padding to a group so that the resulting dataframe will have
    a number of rows equal to the timesteps constant. Also, stay_id
    column is preserved on the padding rows.
    '''
    group = group[:TIMESTEPS]
    n_rows, n_cols = group.shape
    n_rows = TIMESTEPS - n_rows
    padding = np.zeros((n_rows, n_cols))
    padding = pd.DataFrame(padding, columns=group.columns)
    padding['stay_id'] = group['stay_id'].iloc[0]
    return pd.concat([group, padding], axis=0)


def extract_dataset(output_dir: str = 'dataset', redo: bool = False):
    # create output dir if it does not exist
    # all of the artifacts of this script will be put inside this directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)

    # partition features into days
    # transform the huge events table into a feature table
    ipath = MIMIC4_PATH / 'filtered_events.csv'
    opath = output_dir / 'events_partitioned.csv'
    if redo or not opath.exists():
        partition_rows(ipath, opath)

    # not all feature values has a valid value (some of them have NaN)
    # we use a combination of forward and backward imputation to fill these holes
    ipath = opath
    opath = output_dir / 'events_imputed_pretraining.csv'
    if redo or not opath.exists():
        impute_holes(ipath, opath)

    # in addition to the feature values (dynamic), add the patient's
    # demographic information (static)
    ipath = opath
    opath = output_dir / 'events_with_demographics_pretraining.csv'
    if redo or not opath.exists():
        add_patient_info(ipath, opath)

    # MIMIC-IV contains typographical errors and this will come out as outliers
    # in this step, we remove these outliers
    ipath = opath
    opath = output_dir / 'events_complete_pretraining.csv'
    if redo or not opath.exists():
        transform_outliers(ipath, opath)

    # transform CSV data into a 3D tensor for model pretraining
    ipath = opath
    opath = output_dir / 'pretraining_data'
    if redo or not opath.exists():
        transform_to_tensor(ipath, opath)


if __name__ == '__main__':
    fire.Fire(extract_dataset)
