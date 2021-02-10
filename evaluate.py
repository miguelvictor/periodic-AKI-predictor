from models import PredictiveModel2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import fire
import numpy as np
import os
import torch


def evaluate(ckpt_path: str, test_path: str):
    # load testing dataset
    data = np.load(test_path)
    data = torch.as_tensor(data, dtype=torch.float32)
    assert data.ndim == 3

    # load model from checkpoint
    model = PredictiveModel2.load_from_checkpoint(ckpt_path)
    model.eval()

    # get model predictions
    x, y = data[:, :, :-1], data[:, :, -1]
    with torch.no_grad():
        y_hat = torch.sigmoid(model(x))

    # create mask so that padding days and the first day predictions
    # are not included in the evaluation
    mask = x.bool().any(dim=-1)
    mask[:, 0] = False

    y_true = y[mask]
    y_pred = y_hat[mask]

    # calculate statistics
    cm = confusion_matrix(y_true, np.around(y_pred))
    acc = accuracy_score(y_true, np.around(y_pred))
    score = roc_auc_score(y_true, y_pred)
    report = classification_report(y_true, np.around(y_pred))

    print(f'\n[INFO] Evaluation Results: {model.__class__.__name__}')
    print(cm)
    print(f'Accuracy: {acc:.4%}')
    print(f'ROC-AUC Score: {score:.4%}')
    print(report)
    print('=' * 40)


def main(ckpt_path: str, test_path: str):
    # check existence of checkpoint file
    assert os.path.isfile(ckpt_path), "Checkpoint file does not exist."

    # check existence of serialized testing_set file
    assert os.path.isfile(test_path), "Testing dataset file does not exist."

    # evaluate model on the testing set
    evaluate(ckpt_path, test_path)


if __name__ == '__main__':
    fire.Fire(main)
