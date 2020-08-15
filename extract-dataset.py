from pathlib import Path

import fire
import numpy as np
import pandas as pd
import re
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

    # convert height (inches to cm)
    mask = (df['itemid'] == 226707) | (df['itemid'] == 1394)
    df.loc[mask, 'valuenum'] = df[mask]['valuenum'] * 2.54

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

    # save result
    df.to_csv(output_path, index=False)


def impute_holes(input_path, output_path):
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # collect all feature keys
    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}.keys()

    # fill NaN values with the average feature value (only for the current ICU stay)
    # ICU stays with NaN average values are dropped
    icustay_ids = pd.unique(df['icustay_id'])
    for icustay_id in icustay_ids:
        # get mask for the current icu stay
        stay_id_mask = df['icustay_id'] == icustay_id

        # there are ICU stays that even though its los >= 3
        # the actual measurements done in labevents or chartevents are fewer than that
        # so we drop them here
        if df[stay_id_mask].shape[0] < 3:
            print(f'WARN - Will drop ICU stay: id={icustay_id} (creatinine)')
            df = df[~stay_id_mask]
            continue

        # drop ICU stays with no creatinine levels
        # after the first 48 hours
        if not np.isfinite(df[stay_id_mask]['creatinine'].values[2:]).any():
            print(f'WARN - Will drop ICU stay: id={icustay_id} (creatinine)')
            df = df[~stay_id_mask]
            continue

        nan_index = get_nan_index(df[stay_id_mask]['creatinine'])
        # drop ICU stays with no creatinine levels
        # at the third day
        if nan_index == 2:
            print(f'WARN - Will drop ICU stay: id={icustay_id} (creatinine)')
            df = df[~stay_id_mask]
            continue
        # drop ICU stay days (and onwards) with no creatinine levels defined
        else:
            print(f'WARN - Will drop days({nan_index}) of id={icustay_id}')
            nan_indices = df[stay_id_mask].index[nan_index:]
            df = df.drop(nan_indices)

        # fill feature missing values with the mean value
        # of the ICU stay, dropping ICU stays with missing values
        df = fill_nas_or_drop(df, icustay_id, features)

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
    ethnicity_mapping = dict(
        zip(admissions['hadm_id'], admissions['ethnicity']))

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

    # add patient's ethnicity (black or not)
    df['ethnicity'] = df['hadm_id'].apply(lambda x: ethnicity_mapping[x])
    df['black'] = df['ethnicity'].str.contains(
        r'.*black.*', flags=re.IGNORECASE).astype('int')

    # drop unneeded columns
    del df['dob']
    del df['yob']
    del df['admittime']
    del df['admityear']
    del df['ethnicity']

    # save result
    df.to_csv(output_path, index=False)


def add_aki_labels(input_path, output_path):
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    icustay_ids = pd.unique(df['icustay_id'])
    for icustay_id in icustay_ids:
        # get auxiliary variables
        stay_id_mask = df['icustay_id'] == icustay_id
        black = df[stay_id_mask]['black'][0]
        age = df[stay_id_mask]['age'][0]
        gender = df[stay_id_mask]['gender'][0]

        # get difference of creatinine levels
        scr = df[stay_id_mask]['creatinine']
        diffs = scr[1:].values - scr[:-1].values

        # drop ICU stays with AKIs for the first 48 hours
        if (
            has_aki(diff=diffs[0])
            or has_aki(scr=scr[0], black=black, age=age, gender=gender)
            or has_aki(scr=scr[1], black=black, age=age, gender=gender)
        ):
            print(f'WARN - Will drop ICU stay: id={icustay_id} (AKI 48 hours)')
            df = df[~stay_id_mask]
            continue

        # we do next-day AKI prediction
        # use the 3rd day's creatinine level to get the AKI label of day 2 data
        aki1 = pd.Series(diffs[1:]).apply(lambda x: has_aki(diff=x))
        aki2 = scr[2:].apply(lambda x: has_aki(
            scr=x, black=black, age=age, gender=gender))
        aki = (aki1 | aki2).astype('int').values.tolist()

        # drop last day values
        last_day_index = df[stay_id_mask].index[-1]
        df = df.drop(last_day_index)

        # assign aki labels
        df.loc[stay_id_mask, 'aki'] = pd.Series([0] + aki)

    # save results
    df.to_csv(output_path, index=False)


def has_aki(diff=None, scr=None, black=None, age=None, gender=None):
    # KDIGO criteria no. 1
    # Increase in SCr by >= 0.3 mg/dl (>= 26.5 lmol/l) within 48 hours
    if diff is not None:
        return diff >= 0.3

    # KDIGO criteria no. 2
    # increase in SCr to ≥1.5 times baseline, which is known or
    # presumed to have occurred within the prior 7 days
    if scr is not None:
        assert black is not None
        assert age is not None
        assert gender is not None

        baseline = get_baseline(black=black, age=age, gender=gender)
        return scr >= 1.5 * baseline

    # KDIGO criteria no. 3
    # Urine volume <0.5 ml/kg/h for 6 hours
    # not included since urine output data is scarce in MIMIC-III dataset

    raise AssertionError('ERROR - Should pass diff OR scr')


def get_baseline(*, black, age, gender):
    if 20 <= age <= 24:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.5 if gender == 1 else 1.2
        else:
            # other males: 1.3, other females: 1.0
            return 1.3 if gender == 1 else 1.0

    if 25 <= age <= 29:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.5 if gender == 1 else 1.1
        else:
            # other males: 1.3, other females: 1.0
            return 1.2 if gender == 1 else 1.0

    if 30 <= age <= 39:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.4 if gender == 1 else 1.1
        else:
            # other males: 1.3, other females: 1.0
            return 1.2 if gender == 1 else 0.9

    if 40 <= age <= 54:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.3 if gender == 1 else 1.0
        else:
            # other males: 1.3, other females: 1.0
            return 1.1 if gender == 1 else 0.9

    # for ages > 65
    if black == 1:
        # black males: 1.5, black females: 1.2
        return 1.2 if gender == 1 else 0.9
    else:
        # other males: 1.3, other females: 1.0
        return 1.0 if gender == 1 else 0.8


def get_nan_index(series):
    result = ~np.isfinite(series)
    for i, x in enumerate(result[2:]):
        if x:
            return i + 2

    return -1


def extract_dataset(output_dir='dataset'):
    # create output dir if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)

    # partition features into days
    ipath = output_dir / 'filtered_events.csv'
    opath = output_dir / 'events_partitioned.csv'
    partition_rows(ipath, opath)

    # fill empty holes with median values
    ipath = opath
    opath = output_dir / 'events_imputed.csv'
    impute_holes(ipath, opath)

    # add patient info
    ipath = opath
    opath = output_dir / 'events_with_demographics.csv'
    add_patient_info(ipath, opath)

    # add AKI labels
    ipath = opath
    opath = output_dir / 'events_complete.csv'
    add_aki_labels(ipath, opath)


if __name__ == '__main__':
    fire.Fire(extract_dataset)
