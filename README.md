# Introduction

MIMIC predictors with pretraining.

# Preparation

Go to Google BigQuery and run the four queries located in the `pg_scripts` directory. Also, save the query results as csv files in a directory named `mimic4` which is also inside a directory named `databases`.

- `pg_filter_admissions.sql`'s query results should be saved as `filtered_admissions.csv`
- `pg_filter_events.sql`'s query results should be saved as `filtered_events.csv`
- `pg_filter_icustays.sql`'s query results should be saved as `filtered_icustays.csv`
- `pg_filter_patients.sql`'s query results should be saved as `filtered_patients.csv`

# Pretraining

To prepare the datasets needed for both pretraining and finetuning, run the following script:

```
python extract-dataset.py
```

To prepare the dataset for pretraining, run the command below.

```
python prepare-pretraining-data.py
```

To start pretraining, run the command below. Hyperparameters can also be set as arguments to this command (can also be omitted and the defaults will be used).

```
python pretrain.py --n_days=8 --n_features=16 --n_layers=16 --n_head=4 --d_model=1024 --learning_rate=0.001 --gpus=1
```

# Finetuning

To prepare the dataset for finetuning, run the command below. This expects a csv file named `events_complete.csv` in a directory named `dataset`. These can be changed by passing the arguments `name` and `dataset_dir` respectively.

```
python prepare-finetuning-data.py --name=events_complete.csv --dataset_dir=dataset
```

To finetune the model after pretraining, simply provide the pretrained checkpoint. The `gpus` argument can also be provided if you have GPU (omit argument if no cuda-enabled gpu). More options are available (check [here](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html)).

```
python finetune.py --path=<path-to-ckpt-file> --gpus=1
```

To train the model from scratch (without pretraining), do not provide the path argument.

```
python finetune.py --path=<path-to-ckpt-file> --gpus=1
```

# Evaluation

To evaluate a saved checkpoint, run the command below. This command expects a file named `finetuning_testing.npy` in the current directory. To point to other location, pass the `test_path` argument.

```
python evaluate.py <path-to-ckpt-file>
```
