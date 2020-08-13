from pathlib import Path

import numpy as np
import pandas as pd
import fire

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
    # below features are used for determining AKI
    'serum creatinine': [51081],
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
OUTPUTEVENTS_FEATURES = {}


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
        # get mask for the current icu stay
        stay_id_mask = df['icustay_id'] == icustay_id

        for feature in features.keys():
            # compute the average value of the current feature
            # for the current ICU stay
            mean = np.nanmean(df[stay_id_mask][feature])

            # get mask for all of the NaN values
            nan_mask = df[feature].isna()

            if mean == np.nan:
                # drop ICU stay rows (each ICU stay has multiple days)
                # if the computed average is still NaN
                df = df[~stay_id_mask]
            else:
                # fill NaN values of the current feature for the current ICU stay
                # using the average computed above
                df.loc[stay_id_mask & df[nan_mask], feature] = mean

    # save result
    df.to_csv(output_path, index=False)


def add_patient_info(input_path, output_path):
    admissions_path = MIMIC_PATH / 'ADMISSIONS.csv'
    admissions = pd.read_csv(admissions_path)
    admissions.columns = map(str.upper, admissions.columns)

    patients_path = MIMIC_PATH / 'PATIENTS.csv'
    patients = pd.read_csv(patients_path)
    patients.columns = map(str.upper, patients.columns)

    df = pd.read_csv(input_path)
    df.columns = map(str.upper, df.columns)

    # get auxiliary features
    admittime_dict = dict(zip(admissions['HADM_ID'], admissions['ADMITTIME']))
    dob_dict = dict(zip(patients['SUBJECT_ID'], patients['DOB']))
    gender_dict = dict(zip(patients['SUBJECT_ID'], patients['GENDER']))

    # retrieve admission ID from HADMID_DAY
    df['HADM_ID'] = df['HADMID_DAY'].str.split('_').apply(lambda x: x[0])
    df['HADM_ID'] = df['HADM_ID'].astype('int')

    # compute patient's age
    df['DOB'] = df['SUBJECT_ID'].apply(lambda x: dob_dict[x])
    df['YOB'] = df['DOB'].str.split('-').apply(lambda x: x[0]).astype('int')
    df['ADMITTIME'] = df['HADM_ID'].apply(lambda x: admittime_dict[x])
    df['ADMITYEAR'] = df['ADMITTIME'].str.split(
        '-').apply(lambda x: x[0]).astype('int')
    df['AGE'] = df['ADMITYEAR'].subtract(df['YOB'])

    # drop patients with age > 89
    criteria = df['AGE'] <= 89
    df = df[criteria]

    # add patient's gender
    df['GENDER'] = df['SUBJECT_ID'].apply(lambda x: gender_dict[x])
    df['GENDER'] = (df['GENDER'] == 'M').astype('int')

    # add patient's BMI group
    df['BMI'] = df['WEIGHT'] / (df['HEIGHT'] / 100) ** 2
    df['BMI_GROUP'] = 1
    df.loc[df['BMI'] >= 18.5, 'BMI_GROUP'] = 2
    df.loc[df['BMI'] >= 24, 'BMI_GROUP'] = 3
    df.loc[df['BMI'] >= 28, 'BMI_GROUP'] = 4

    # drop unneeded columns
    del df['DOB']
    del df['YOB']
    del df['ADMITTIME']
    del df['ADMITYEAR']
    del df['BMI']

    # save result
    df.to_csv(output_path, index=False)


def extract_dataset(output_dir='dataset'):
    # create output dir if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)

    # partition features into days
    ipath = output_dir / 'filtered_events.csv'
    opath = output_dir / 'events_partitioned.csv'
    partition_rows(ipath, opath)

    # add patient info
    ipath = opath
    opath = output_dir / 'events_with_demographics.csv'
    add_patient_info(ipath, opath)


if __name__ == '__main__':
    fire.Fire(extract_dataset)
