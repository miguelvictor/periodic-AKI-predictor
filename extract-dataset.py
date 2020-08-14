from pathlib import Path

import numpy as np
import pandas as pd
import fire
import warnings

DB_PATH = Path('databases')
MIMIC_PATH = DB_PATH / 'mimic3'
LABEVENTS_FEATURES = {
    'bicarbonate': [50882],  # mEq/L
    'chloride': [50902],
    'creatinine': [50912],
    'glucose': [50931],
    'magnesium': [50960],
    'potassium': [50822, 50971],  # mEq/L
    'sodium': [50824, 50983],  # mEq/L == mmol/L
    'BUN': [51006],
    'hemoglobin': [51222],
    'platelets': [51265],
    'WBCs': [51300, 51301],
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


def partition_rows(input_path, output_path, stats_path):
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # extract the day of the event
    df['chartday'] = df['charttime'].astype(
        'str').str.split(' ').apply(lambda x: x[0])

    # group day into a specific ICU stay
    df['icu_day'] = df['icustay_id'].astype('str') + '_' + df['chartday']

    # add feature label column
    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}
    features_reversed = {v2: k for k, v1 in features.items() for v2 in v1}
    df['feature'] = df['itemid'].apply(lambda x: features_reversed[x])

    # save mapping of icu stay ID to patient ID
    icu_subject_mapping = dict(zip(df['icu_day'], df['subject_id']))

    # average all feature values each day
    df = pd.pivot_table(
        df,
        index='icu_day',
        columns='feature',
        values='valuenum',
        fill_value=np.nan,
        aggfunc=np.nanmean,
        dropna=False,
    )

    # insert back information related to the patient (for persistence)
    df['icu_day'] = df.index
    df['icustay_id'] = df['icu_day'].str.split(
        '_').apply(lambda x: x[0]).astype('int')
    df['subject_id'] = df['icu_day'].apply(lambda x: icu_subject_mapping[x])

    # fill NaN values with the average feature value (only for the current ICU stay)
    # ICU stays with NaN average values are dropped
    icustay_ids = pd.unique(df['icustay_id'])
    for icustay_id in icustay_ids:
        df = fill_nas_or_drop(df, icustay_id, features.keys())

    # save result
    df.to_csv(output_path, index=False)


def fill_nas_or_drop(df, icustay_id, features):
    # get mask for the current icu stay
    stay_id_mask = df['icustay_id'] == icustay_id

    for feature in features:
        # compute the average value of the current feature
        # for the current ICU stay
        # this will generate warnings for ICU stays that doesn't
        # have features for the whole ICU stay span, so suppress that
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(df[stay_id_mask][feature])

        # drop ICU stay rows (each ICU stay has multiple days)
        # if the computed average is still NaN
        if not np.isfinite(mean):
            return df[~stay_id_mask]

        # get mask for all of the NaN values
        nan_mask = df[feature].isna()

        # fill NaN values of the current feature for the current ICU stay
        # using the average computed above
        df.loc[stay_id_mask & nan_mask, feature] = mean

    return df


def add_patient_info(input_path, output_path):
    admissions_path = MIMIC_PATH / 'admissions.csv'
    admissions = pd.read_csv(admissions_path)
    admissions.columns = map(str.lower, admissions.columns)

    icustays_path = MIMIC_PATH / 'icustays.csv'
    icustays = pd.read_csv(icustays_path)
    icustays.columns = map(str.lower, icustays.columns)

    patients_path = MIMIC_PATH / 'patients.csv'
    patients = pd.read_csv(patients_path)
    patients.columns = map(str.lower, patients.columns)

    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # get auxiliary features
    hadm_id_mapping = dict(zip(icustays['icustay_id'], icustays['hadm_id']))
    admittime_mapping = dict(
        zip(admissions['hadm_id'], admissions['admittime']))
    dob_mapping = dict(zip(patients['subject_id'], patients['dob']))
    gender_mapping = dict(zip(patients['subject_id'], patients['gender']))

    # retrieve admission ID from HADMID_DAY
    df['icustay_id'] = df['icu_day'].str.split('_').apply(lambda x: x[0])
    df['icustay_id'] = df['icustay_id'].astype('int')
    df['hadm_id'] = df['icustay_id'].apply(lambda x: hadm_id_mapping[x])

    # compute patient's age
    df['dob'] = df['subject_id'].apply(lambda x: dob_mapping[x])
    df['yob'] = df['dob'].str.split('-').apply(lambda x: x[0]).astype('int')
    df['admittime'] = df['hadm_id'].apply(lambda x: admittime_mapping[x])
    df['admityear'] = df['admittime'].str.split(
        '-').apply(lambda x: x[0]).astype('int')
    df['age'] = df['admityear'].subtract(df['yob'])

    # set max age to 90
    df.loc[df['age'] > 89, 'age'] = 90

    # add patient's gender
    df['gender'] = df['subject_id'].apply(lambda x: gender_mapping[x])
    df['gender'] = (df['gender'] == 'M').astype('int')

    # drop unneeded columns
    del df['dob']
    del df['yob']
    del df['admittime']
    del df['admityear']

    # save result
    df.to_csv(output_path, index=False)


def extract_dataset(output_dir='dataset'):
    # create output dir if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)
    stats_path = output_dir / 'stats.txt'

    # partition features into days
    ipath = output_dir / 'filtered_events.csv'
    opath = output_dir / 'events_partitioned.csv'
    partition_rows(ipath, opath, stats_path)

    # add patient info
    ipath = opath
    opath = output_dir / 'events_with_demographics.csv'
    add_patient_info(ipath, opath)


if __name__ == '__main__':
    fire.Fire(extract_dataset)
