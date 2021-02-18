from models import PredictiveModel2
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from typing import List

import fire
import numpy as np
import os
import torch


def evaluate(ckpt_path: str, test_path: str = 'finetuning_testing.npy'):
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

    print('=' * 60)
    print(f'Accuracy: {acc:.4%}')
    print(f'ROC-AUC Score: {score:.4%}')
    print(format_confusion_matrix(cm))
    print(report)
    print('=' * 60)


def format_confusion_matrix(cm: List[List[int]]):
    x = PrettyTable()
    x.field_names = ["", "Model (-)", "Model (+)"]
    x.add_row(["Actual (-)", cm[0][0], cm[0][1]])
    x.add_row(["Actual (+)", cm[1][0], cm[1][1]])
    return x


def main(ckpt_path: str, test_path: str):
    # check existence of checkpoint file
    assert os.path.isfile(ckpt_path), "Checkpoint file does not exist."

    # check existence of serialized testing_set file
    assert os.path.isfile(test_path), "Testing dataset file does not exist."

    # evaluate model on the testing set
    evaluate(ckpt_path, test_path)


if __name__ == '__main__':
    fire.Fire(main)
