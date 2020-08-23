## Getting MIMIC4

[MIMIC4](https://mimic-iv.mit.edu/) contains many table but only `chartevents`, `labevents`, `patients`, `admissions`, and `icustays` table are used. Also, not all of the rows of the tables mentioned are used. In order to download only the required tables of this project, the following SQL scripts should be run within [Google’s BigQuery](https://mimic-iv.mit.edu/docs/access/bigquery/) and the results should be downloaded as csv. The following scripts are located inside the `pg_scripts` directory.

1. `pg_filter_admissions.sql` filters the table `mimic_core.admissions` saving only the needed admissions which contains the admission time and the ethnicity of a patient.
2. `pg_filter_events.sql` filters the table `mimic_hosp.labevents` and `mimic_icu.chartevents` saving only the the measurements of the features that our model uses.
3. `pg_filter_icustays.sql` filters the table `mimic_icu.icustays` saving only the ICU stays with length of stay greater than or equal to 3 days.
4. `pg_filter_patients.sql` filters the table `mimic_core.patients` saving only patients with ICU stay length greater than or equal to 3 days.

## Dataset Preprocessing

The dataset preprocessing part consists of 5 major parts:

1. `partition_rows` takes care of grouping the feature values into ICU stay days. If a feature value has multiple measuremeants within the same day, the values are averaged. There would be days where there are no feature values observed and they will be `NaN` after this operation. Also, ICU stays with length of stay greater than or equal to 3 days isn't guaranteed to have 3 days in this operation (caused by missing feature values for the whole ICU stay span).

2. `impute_holes` fills in the `NaN` values in the dataframe created by `partition_rows`. After the following checks are performed, this operation will impute `NaN` values using forward/backward imputation.

   - Since the column `los` of the `mimic_icu.icustays` doesn't guarantee that an ICU stay will have measurements for the required features in 3 days span, `impute_holes` will drop ICU stays with no measured values in 3 days span.
   - ICU stays with no `creatinine` for all of the days after the first 2 days are dropped (since there's no way to tell if they have AKI for these days; ICU stays with no `creatinine` for the first 2 days are assumed to not have AKI)
   - ICU stays with no `creatinine` on the 3rd day are dropped.
   - If an ICU stay contains `creatinine` for the first 3 days and doesn't contain `creatinine` on 4th day onwards, then the ICU stay's data on 4th day onwards are discarded.

3. `add_patient_info` adds patient's demographics data (age, gender, and ethnicity).

4. `add_aki_labels` adds the appropriate next-day AKI labels except ICU stays with AKI detected for the first 2 days (since they are dropped). The AKI definition used in this project is the KDIGO criteria which is detailed [here](https://kdigo.org/wp-content/uploads/2016/10/KDIGO-2012-AKI-Guideline-English.pdf). Also, Patients with age less than 20 years old are also dropped (since KDIGO criteria doesn't have a baseline for persons with age < 20 years old). Due to the lack of urine output data in MIMIC4 dataset, only the first two criteria are used:

   > Increase in SCr by >= 0.3 mg/dl (>= 26.5 lmol/l) within 48 hours; or

   > Increase in SCr to >= 1.5 times baseline, which is known or presumed to have occurred within the prior 7 days

5. `transform_outliers` checks the minimum and maximum values of each of the features to check if they're valid or not (MIMIC4 dataset sometimes contains errors caused by typos). Features with values lower than the lower bound or higher than the upper bound are replaced with the defined lower and upper bound. The lower and upper bound of a feature are defined by the following equations:

$$
\text{lower bound} = \mu - 6\sigma\\
\text{upper bound} = \mu + 6\sigma\\
\text{where}\ \mu\ \text{and}\ \sigma\ \text{is the mean and std respectively}
$$

## Oversampling methods

Since the [dataset](https://github.com/miggymigz/periodic-AKI-predictor/blob/master/dataset-visualization.ipynb) is highly imbalanced, two methods are considered in this work: oversampling by duplicating and using Generative Adversarial Networks (GAN) to generate synthetic data for the minority class (courtesy of [this paper](https://arxiv.org/pdf/1901.02514.pdf)).

## Evaluation Metrics

The primary evaluation metric used in this project is the ROC AUC score since it is a standard technique for summarizing classifier performance over a range of tradeoffs between true positive and false positive error rates. In addition to this, accuracy, precision, recall and f1-score (using a threshold of 0.5; uses `np.around` in the code) are also added to give a rough sense of the overall model performance.

To calculate the mentioned metrics, instead of comparing all of the model's predictions to the ground truth, only the last day prediction is compared to the last day's ground truth. This is analogous to text generation models (like GPT-2) in which we can see the model's performance with its next token prediction (the next token is the most probable token given the previous tokens; in this work, this can be interpreted as what is the next-day AKI probability of the patient given the previous days' data).

In addition to the above mentioned metrics, the accuracy of the model's ability to predict AKI in advance (2-day window) is also considered. This metric makes sense since the goal of this paper is early prediction of AKI which has the potential to save the lives of patients in ICU.
