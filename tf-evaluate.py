from pathlib import Path
from predictor.models import TFAkiBase, TFAkiLstm, TFAkiGpt2
from predictor.models.tf_mlp import TFMLPBase 
from predictor.utils import convert_preds, early_prediction_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import fire
import numpy as np
import os
import tensorflow as tf

TIMESTEPS = 8
N_FEATURES = 16


def evaluate(
    ckpt_dir: str = 'saved_models',
    testing: str = 'matrix_testing.npy',
    dataset_dir: str = 'dataset',
):
    '''
    Tests the serialized models inside the `ckpt_dir` directory.
    `ckpt_directory` should contain ONLY the model's weights and not the
    entire model (saved using torch.save(model.state_dict())). Filenames
    should follow the pattern {architecture}_e{epochs}_l{n_layers}.pt.

    Parameters:
    ckpt_dir: The location of the serialized model state dicts
    testing: The testing data to be used (a numpy 3d-array serialized using np.save)
    dataset_dir: The directory that contains the `testing` numpy array
    '''
    ckpt_dir = Path(ckpt_dir)
    assert ckpt_dir.exists(), \
        f'"{ckpt_dir}" is empty. Train models first.'

    dataset_path = Path(dataset_dir)
    testing_path = dataset_path / testing
    assert testing_path.exists(), f'{testing} does not exist'

    test_matrix = np.load(testing_path).astype(np.float32)
    test_x = test_matrix[:, :, :-1]
    test_y = test_matrix[:, :, -1:]

    test_mask = tf.reduce_any(test_x!=0,axis=-1)
    test_x = tf.boolean_mask(test_x,test_mask)
    test_y = tf.boolean_mask(test_y,test_mask)

    for model in get_models(ckpt_dir):
        # get model's predictions
        outputs = model(test_x)
        y_hat = outputs[0] if isinstance(outputs, tuple) else outputs
        #print(test_y.shape,y_hat.shape)
        # get model's early prediction score
        escore, stats = early_prediction_score(test_y, np.around(y_hat))

        # convert predictions to last-day AKI predictions
        #y, y_hat = convert_preds(test_x, test_y, y_hat)

        y = test_y
        cm = confusion_matrix(y, np.around(y_hat))
        acc = accuracy_score(y, np.around(y_hat))
        score = roc_auc_score(y, y_hat)
        report = classification_report(y, np.around(y_hat))

        print(f'\n[INFO] Evaluation Results: {model.__class__.__name__}')
        print(cm)
        print(f'Accuracy: {acc:.4%}')
        print(f'ROC-AUC Score: {score:.4%}')
        print(f'Early Detection Accuracy: {escore:.4%}, {stats}\n')
        print(report)
        print('=' * 40)


def get_models(ckpt_dir: Path):
    for dname in os.listdir(ckpt_dir):
        # get the model's architecture
        # from the filename of its trained weights
        architecture, _ = dname.split('_', 1)

        # load model's trained weights
        model_weights_path = ckpt_dir / dname / architecture

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
    if architecture=='mlp':
        return TFMLPBase()

    raise AssertionError(f'Unknown architecture "{architecture}"')


if __name__ == '__main__':
    fire.Fire(evaluate)
