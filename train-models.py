from pathlib import Path
from predictor.models import AkiLstm
from predictor.data import Mimic3Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import fire
import logging
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(
    filename='train-models.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger('default')


def train_models(epochs: int = 1):
    dataset_dir = Path('dataset')
    dataset = Mimic3Dataset(dataset_dir / 'events_complete.csv')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128,
        shuffle=True, num_workers=4
    )

    model = AkiLstm(timesteps=8, n_features=16)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_obj = torch.nn.BCELoss(reduction='mean')

    for i in tqdm(range(epochs)):
        for j, (x, y) in tqdm(enumerate(dataloader)):
            # clear accumulated gradients
            model.zero_grad()

            # get model's outputs
            y_hat = model(x)

            # compute loss (excluding day 1 and padding days)
            # and compute gradients
            mask = x.byte().any(dim=-1).type(torch.bool)
            mask[:, 0] = False
            loss = loss_obj(y_hat[mask], y[mask])
            loss.backward()

            # update model's parameters
            optimizer.step()

            with torch.no_grad():
                score = roc_auc_score(y[mask], y_hat[mask])
                logger.info(
                    f'Epoch {i} Batch {j}: roc_auc_score={score} loss={loss}')


if __name__ == '__main__':
    fire.Fire(train_models)
