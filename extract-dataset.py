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


def filter_chunk(
    *,
    chunk,
    item_ids,
    output_path,
    nonnull_columns,
    columns_to_save,
    first=False,
):
    # make sure headers are upper-cased
    chunk.columns = map(str.upper, chunk.columns)

    # drop rows that contains null values in the VALUE column
    # drop rows that is not relevant
    mask = chunk['ITEMID'].isin(item_ids)
    chunk = chunk[mask].dropna(axis=0, how='any', subset=nonnull_columns)

    # make sure HADM_ID is int
    chunk = chunk.astype({'HADM_ID': 'int32'})

    # convert height (inches to cm)
    mask = chunk['ITEMID'].isin([226707, 1394])
    chunk.loc[mask, 'VALUENUM'] = chunk.loc[mask, 'VALUENUM'] * 2.54

    if first:
        chunk.to_csv(output_path, index=False, columns=columns_to_save)
    else:
        chunk.to_csv(
            output_path,
            index=False,
            columns=columns_to_save,
            header=None,
            mode='a',
        )


def filter_relevant_rows(output_path, chunksize=10_000_000):
    labevents_path = MIMIC_PATH / 'LABEVENTS.csv'
    chartevent_path = MIMIC_PATH / 'CHARTEVENTS.csv'

    # define columns that should be non-null
    # also the columns to save for this preprocessing step
    nonnull_columns = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE']
    columns_to_save = nonnull_columns + ['VALUENUM', 'VALUEUOM']

    # concatenate all of the item IDs that is relevant
    labevents_item_ids = [
        itemid
        for ids in LABEVENTS_FEATURES.values()
        for itemid in ids
    ]
    chartevents_item_ids = [
        itemid
        for ids in CHARTEVENTS_FEATURES.values()
        for itemid in ids
    ]

    labevents_iterator = pd.read_csv(
        labevents_path, iterator=True, chunksize=chunksize)
    chartevents_iterator = pd.read_csv(
        chartevent_path, iterator=True, chunksize=chunksize, low_memory=False)
    first = True

    for chunk in labevents_iterator:
        filter_chunk(
            chunk=chunk,
            item_ids=labevents_item_ids,
            output_path=output_path,
            nonnull_columns=nonnull_columns,
            columns_to_save=columns_to_save,
            first=first,
        )
        first = False

    for chunk in chartevents_iterator:
        filter_chunk(
            chunk=chunk,
            item_ids=chartevents_item_ids,
            output_path=output_path,
            nonnull_columns=nonnull_columns,
            columns_to_save=columns_to_save,
            first=False,
        )


def partition_rows(input_path, output_path):
    df = pd.read_csv(input_path)

    # extract the day of the event
    df['CHARTDAY'] = df['CHARTTIME'].astype(
        'str').str.split(' ').apply(lambda x: x[0])

    # group day into a specific admission
    df['HADMID_DAY'] = df['HADM_ID'].astype('str') + '_' + df['CHARTDAY']

    # add feature label column
    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}
    features_reversed = {v2: k for k, v1 in features.items() for v2 in v1}
    df['FEATURE'] = df['ITEMID'].apply(lambda x: features_reversed[x])

    # save mapping of admission ID to patient ID
    hadm_dict = dict(zip(df['HADMID_DAY'], df['SUBJECT_ID']))

    # average all feature values each day
    df = pd.pivot_table(
        df,
        index='HADMID_DAY',
        columns='FEATURE',
        values='VALUENUM',
        fill_value=np.nan,
        aggfunc=np.nanmean,
        dropna=False,
    )

    # fill NA values with mean values and round to 4 decimal places
    for column in df.columns:
        column_mean = df[column].mean(skipna=True)
        print(f'Column "{column}" mean: {column_mean}')

        df[column] = df[column].fillna(column_mean)
        df[column] = df[column].round(4)

    # insert back information related to the patient (for persistence)
    df['HADMID_DAY'] = df.index
    df['SUBJECT_ID'] = df['HADMID_DAY'].apply(lambda x: hadm_dict[x])

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

    # filter out relevant rows in LABEVENTS
    opath = output_dir / 'LABEVENTS_filtered.csv'
    filter_relevant_rows(opath)

    # partition features into days
    ipath = opath
    opath = output_dir / 'LABEVENTS_partitioned.csv'
    partition_rows(ipath, opath)

    # add patient info
    ipath = opath
    opath = output_dir / 'LABEVENTS_with_demographics.csv'
    add_patient_info(ipath, opath)


if __name__ == '__main__':
    fire.Fire(extract_dataset)
