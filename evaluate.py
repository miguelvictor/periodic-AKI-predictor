from pathlib import Path
from predictor.models import AkiLstm
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
    name: str = 'e1_lstm.pt',
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

    model = AkiLstm(timesteps=8, n_features=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_hat = model(x)
        mask = get_mask_for(x)

        cm = confusion_matrix(y[mask], torch.round(y_hat[mask]))
        acc = accuracy_score(y[mask], torch.round(y_hat[mask]))
        score = roc_auc_score(y[mask], y_hat[mask])
        report = classification_report(y[mask], torch.round(y_hat[mask]))

        print(f'\n[INFO] Model evaluation results')
        print(cm)
        print(f'Accuracy: {acc:.4%}')
        print(f'ROC AUC SCORE: {score:.4%}')
        print(report)
        print('=' * 40)


def get_mask_for(x):
    # exclude day 1 and padding days
    mask = x.byte().any(dim=-1).type(torch.bool)
    mask[:, 0] = False
    return mask


if __name__ == '__main__':
    fire.Fire(evaluate)
