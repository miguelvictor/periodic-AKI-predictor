from pathlib import Path
from predictor.models import AkiLstm
from predictor.utils import get_mask_for, convert_preds
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import fire
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# setup logging stuff
logging.basicConfig(
    filename='train-models.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger('default')

# set random seed (for reproducibility)
np.random.seed(7)
torch.manual_seed(7)

# constants
TIMESTEPS = 8
N_FEATURES = 16


def train_models(
    epochs: int = 1,
    batch_size: int = 256,
    lr: int = 0.0001,
    dataset_dir: str = 'dataset',
    checkpoint_path: str = 'saved_models',
    training: str = 'matrix_training.npy',
    validation: str = 'matrix_validation.npy',
):
    # verify training and validation data exist
    dataset_dir = Path(dataset_dir)
    training_path = dataset_dir / training
    val_path = dataset_dir / validation
    assert training_path.exists(), f'{training} does not exist'
    assert val_path.exists(), f'{validation} does not exist'

    # load training and validation data
    # also, separate the labels from the matrix
    training_matrix = np.load(training_path)
    training_x = torch.tensor(training_matrix[:, :, :-1], dtype=torch.float32)
    training_y = torch.tensor(training_matrix[:, :, -1:], dtype=torch.float32)
    val_matrix = np.load(val_path)
    val_x = torch.tensor(val_matrix[:, :, :-1], dtype=torch.float32)
    val_y = torch.tensor(val_matrix[:, :, -1:], dtype=torch.float32)

    # create a dataset out of the loaded tensors
    dataset = TensorDataset(training_x, training_y)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=4
    )

    # check CUDA availability
    # use it if it is available, warn if absent
    is_available = torch.cuda.is_available()
    device = torch.device('cuda' if is_available else 'cpu')
    if not is_available:
        logger.warning('CUDA is not available. Training will be slow.')

    # configure summary writer for tensorboard visualization
    # this outputs to ./runs/ directory by default
    writer = SummaryWriter(comment=f'_e{epochs}_lr{lr:.0e}')

    # define model architecture and hyperparameters
    model = AkiLstm(timesteps=TIMESTEPS, n_features=N_FEATURES, n_layers=2)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_obj = torch.nn.BCELoss(reduction='mean')

    # print model's architecture
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'No. of trainable parameters: {n_params}')
    logger.info(str(model))

    # start of model training
    for i in range(1, epochs+1):
        # accumulate loss, accuracy, and roc-auc-score
        # for every batch of training
        e_losses = []
        e_accs = []
        e_scores = []

        # use a fancy progressbar to track progress of training
        pbar = tqdm(dataloader)
        pbar.set_description(f'Epoch {i}/{epochs}')

        # start of model training (per batch)
        for x, y in pbar:
            # transfer current batch to GPU
            x, y = x.to(device), y.to(device)

            # set model to training mode
            # also, zero out gradient buffers
            model.train()
            model.zero_grad()

            # compute loss
            y_hat, _ = model(x)
            mask = get_mask_for(x)
            loss = loss_obj(y_hat[mask], y[mask])

            # compute gradients and update model's parameters
            loss.backward()
            optimizer.step()

            # compute accuracy and roc_auc_score for the current batch
            # to be displayed when the current epoch ends.
            # sklearn utility functions operates on tensors on cpu
            # so they are moved as necessary.
            with torch.no_grad():
                # convert y and y_hat into 1d array that contains
                # only the last day prediction
                y, y_hat = convert_preds(x, y, y_hat)

                batch_loss = loss.item()
                batch_acc = accuracy_score(y, torch.round(y_hat))
                batch_score = roc_auc_score(y, y_hat)

                e_losses.append(batch_loss)
                e_accs.append(batch_acc)
                e_scores.append(batch_score)

        # log training statistics after every epoch
        train_loss = torch.tensor(e_losses).mean()
        train_acc = torch.tensor(e_accs).mean()
        train_score = torch.tensor(e_scores).mean()

        # compute statistics with respect to the validation set
        with torch.no_grad():
            # set model to evaluation mode
            # move validation set to GPU (if available)
            model.eval()
            val_x, val_y = val_x.to(device), val_y.to(device)

            # predict and compute loss
            val_y_hat, _ = model(val_x)
            mask = get_mask_for(val_x)
            val_loss = loss_obj(val_y_hat[mask], val_y[mask]).item()

            # convert y and y_hat into 1d array that contains
            # only the last day prediction
            y, y_hat = convert_preds(val_x, val_y, val_y_hat)
            val_acc = accuracy_score(y, torch.round(y_hat))
            val_score = roc_auc_score(y, y_hat)

        # write training statistics to tensorboard summary writer
        # for later visualization
        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Loss/val', val_loss, i)
        writer.add_scalar('Accuracy/train', train_acc, i)
        writer.add_scalar('Accuracy/val', val_acc, i)
        writer.add_scalar('ROC AUC Score/train', train_score, i)
        writer.add_scalar('ROC AUC Score/val', val_score, i)

        # print training and validation statistics
        # on both stdout and logfile
        stats_str = f'acc={train_acc:.4f} val_acc={val_acc:.4f} ' + \
            f'roc_auc_score={train_score:.4f} val_roc_auc_score={val_score:.4f} ' + \
            f'loss={train_loss:.4f} val_loss={val_loss:.4f}'
        print(stats_str)
        logger.info(f'Epoch {i}/{epochs}: {stats_str}')

    # ensure checkpoint directory exists
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=False, exist_ok=True)

    # save model for later use
    model_path = checkpoint_path / f'e{epochs}_lr{lr:.0e}_lstm.pt'
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    fire.Fire(train_models)
