from pathlib import Path
from predictor.models import AkiLstm
from predictor.utils import convert_preds
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import fire
import numpy as np
import torch


def evaluate(
    name: str = 'e1_lr1e-04_lstm.pt',
    checkpoint_dir: str = 'saved_models',
    testing: str = 'matrix_testing.npy',
    dataset_dir: str = 'dataset',
):
    checkpoint_path = Path(checkpoint_dir)
    model_path = checkpoint_path / name
    assert model_path.exists(), f'{name} does not exist. Train model first.'

    dataset_path = Path(dataset_dir)
    testing_path = dataset_path / testing
    assert testing_path.exists(), f'{testing} does not exist'

    test_matrix = np.load(testing_path)
    x = torch.tensor(test_matrix[:, :, :-1], dtype=torch.float32)
    y = torch.tensor(test_matrix[:, :, -1:], dtype=torch.float32)

    model = AkiLstm(timesteps=8, n_features=16, n_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_hat, _ = model(x)
        y, y_hat = convert_preds(x, y, y_hat)

        cm = confusion_matrix(y, torch.round(y_hat))
        acc = accuracy_score(y, torch.round(y_hat))
        score = roc_auc_score(y, y_hat)
        report = classification_report(y, torch.round(y_hat))

        print(f'\n[INFO] Model evaluation results')
        print(cm)
        print(f'Accuracy: {acc:.4%}')
        print(f'ROC AUC SCORE: {score:.4%}')
        print(report)
        print('=' * 40)


if __name__ == '__main__':
    fire.Fire(evaluate)
