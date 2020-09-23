from pathlib import Path
from predictor.models import TFAkiBase, TFAkiLstm, TFAkiGpt2
from predictor.utils import convert_preds, early_prediction_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import fire
import numpy as np

TIMESTEPS = 8
N_FEATURES = 16


def evaluate(
    ckpt_dir: str = 'saved_models',
    dataset_dir: str = 'dataset',
    testing: str = 'matrix_testing.npy',
):
    '''
    After training the models, this script can be invoked to test the models'
    performance on the testing set. 

    Parameters:
    ckpt_dir: The location of the saved model checkpoints.
    dataset_dir: The name of the directory that contains the testing dataset
    testing: The filename of the training dataset to be used (should be a file 
        serialized using np.save and with a shape of [n_samples, timesteps, n_features + 1]
        where 1 refers to the AKI prediction labels)
    '''
    ckpt_dir = Path(ckpt_dir)
    assert ckpt_dir.exists(), \
        f'"{ckpt_dir}" is empty. Train the models first.'

    dataset_path = Path(dataset_dir)
    testing_path = dataset_path / testing
    assert testing_path.exists(), f'{testing} does not exist'

    test_matrix = np.load(testing_path).astype(np.float32)
    test_x = test_matrix[:, :, :-1]
    test_y = test_matrix[:, :, -1:]

    for model in get_models(ckpt_dir):
        # get model's predictions
        outputs = model(test_x)
        y_hat = outputs[0] if isinstance(outputs, tuple) else outputs

        # get model's early prediction score
        escore, stats = early_prediction_score(test_y, np.around(y_hat))

        # convert predictions to last-day AKI predictions
        y, y_hat = convert_preds(test_x, test_y, y_hat)

        # compute evaluation metrics
        cm = confusion_matrix(y, np.around(y_hat))
        acc = accuracy_score(y, np.around(y_hat))
        score = roc_auc_score(y, y_hat)
        report = classification_report(y, np.around(y_hat))

        # report evaluation metrics to stdout
        print(f'\n[INFO] Evaluation Results: {model.__class__.__name__}')
        print(cm)
        print(f'Accuracy: {acc:.4%}')
        print(f'ROC-AUC Score: {score:.4%}')
        print(f'Early Detection Accuracy: {escore:.4%}, {stats}\n')
        print(report)
        print('=' * 40)


def get_models(ckpt_dir: Path):
    # get model names based on what is inside the ckpt_dir
    for f in ckpt_dir.iterdir():
        # files are ignored (only directories are considered)
        if f.is_file():
            continue

        # get the model's architecture
        # from the filename of its trained weights
        architecture, _ = f.name.split('_', 1)

        # load model's trained weights
        model_weights_path = f / architecture

        # create model and restore its trained weights
        # partial is because we don't need the optimizer's state
        model = get_model(architecture)
        model.load_weights(model_weights_path).expect_partial()
        yield model


def get_model(architecture: str):
    if architecture == 'base':
        return TFAkiBase()

    if architecture == 'gpt2':
        return TFAkiGpt2(
            n_heads=2,
            timesteps=TIMESTEPS,
            n_features=N_FEATURES,
        )

    if architecture == 'lstm':
        return TFAkiLstm(
            timesteps=TIMESTEPS,
            n_features=N_FEATURES,
        )

    raise AssertionError(f'Unknown architecture "{architecture}"')


if __name__ == '__main__':
    fire.Fire(evaluate)
