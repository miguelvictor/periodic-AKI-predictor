from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from predictor.models import AkiLstm, AkiGpt2
from predictor.utils import get_mask_for, convert_preds
from predictor.training_args import TrainingArgs

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
    lr: float = 0.0001,
    n_lstm_layers: int = 2,
    n_gpt2_layers: int = 1,
    dataset_dir: str = 'dataset',
    ckpt_dir: str = 'saved_models',
    training: str = 'matrix_training.npy',
    val: str = 'matrix_validation.npy',
):
    '''
    Trains 3 models (LSTM, GPT-2 and CNN) to predict next-day AKI.

    Parameters:
    epochs: For how many epochs we train the models
    batch_size: The batch size to be used during training (the bigger the better)
    lr: The learning rate to be used with Adam optimizer
    n_lstm_layers: The number of LSTM layers to be stacked on top of one another
    n_gpt2_layers: The number of decoder blocks to be used on the model
    dataset_dir: The name of the directory which should contain the 
        training and validation datasets
    ckpt_dir: The name of the directory which the serialized weights of the
        trained models are saved.
    training: The training dataset to be used (should be a file serialized using 
        np.save and with a shape of [n_samples, timesteps, n_features + 1]
        where 1 stands for the AKI prediction labels)
    val: The validation dataset to be used (should be a file serialized using 
        np.save and with a shape of [n_samples, timesteps, n_features + 1]
        where 1 stands for the AKI prediction labels)
    '''
    # verify training and validation data exist
    dataset_dir = Path(dataset_dir)
    train_path = dataset_dir / training
    val_path = dataset_dir / val
    assert train_path.exists(), f'{train} does not exist'
    assert val_path.exists(), f'{val} does not exist'

    # load training and validation data
    # also, separate the labels from the matrix
    train_matrix = np.load(train_path)
    train_x = torch.tensor(train_matrix[:, :, :-1], dtype=torch.float32)
    train_y = torch.tensor(train_matrix[:, :, -1:], dtype=torch.float32)
    val_matrix = np.load(val_path)
    val_x = torch.tensor(val_matrix[:, :, :-1], dtype=torch.float32)
    val_y = torch.tensor(val_matrix[:, :, -1:], dtype=torch.float32)

    # create a dataset out of the loaded tensors
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    train_dl = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
    )
    val_dl = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
    )

    # check CUDA availability
    # use it if it is available, warn if absent
    is_available = torch.cuda.is_available()
    device = torch.device('cuda' if is_available else 'cpu')
    if not is_available:
        logger.warning('CUDA is not available. Training will be slow.')

    # create training args
    args = TrainingArgs(
        epochs=epochs,
        lr=lr,
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        ckpt_dir=Path(ckpt_dir),
    )

    # train LSTM (baseline model)
    args.n_layers = n_lstm_layers
    train('lstm', args)

    # train GPT-2
    args.n_layers = n_gpt2_layers
    train('gpt2', args)


def train(name: str, args: TrainingArgs):
    # extract training arguments
    epochs = args.epochs
    lr = args.lr
    n_layers = args.n_layers
    device = args.device
    train_dl = args.train_dl
    val_dl = args.val_dl
    ckpt_dir = args.ckpt_dir

    # configure summary writer for tensorboard visualization
    # this outputs to ./runs/ directory by default
    writer = get_summary_writer(name=name, epochs=epochs, lr=lr)

    # define model architecture and hyperparameters
    model = get_model(name=name, n_layers=n_layers)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_obj = torch.nn.BCELoss(reduction='mean')

    # print model's architecture
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'No. of trainable parameters: {n_params}')
    logger.info(str(model))

    # start of model training
    for i in range(1, epochs + 1):
        # accumulate loss, accuracy, and roc-auc-score
        # for every batch in the training set
        e_losses = []
        e_accs = []
        e_scores = []

        # use a fancy progressbar to track progress of training
        pbar = tqdm(train_dl)
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
            model.eval()

            # accumulate loss, accuracy, and roc-auc-score
            # for every batch in the validation set
            val_losses = []
            val_accs = []
            val_scores = []

            for val_x, val_y in val_dl:
                # move validation set to GPU (if available)
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

                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_scores.append(val_score)

            # average evaluation metrics
            val_loss = torch.tensor(val_losses).mean()
            val_acc = torch.tensor(val_accs).mean()
            val_score = torch.tensor(val_scores).mean()

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

    # save model for later use
    # ensure checkpoint directory exists
    ckpt_dir.mkdir(parents=False, exist_ok=True)
    model_path = ckpt_dir / f'{name}_e{epochs}_lr{lr:.0e}_lstm.pt'
    torch.save(model.state_dict(), model_path)


def get_model(*, name: str, n_layers: int):
    if name == 'lstm':
        return AkiLstm(
            timesteps=TIMESTEPS,
            n_features=N_FEATURES,
            n_layers=n_layers,
        )

    if name == 'gpt2':
        return AkiGpt2(
            timesteps=TIMESTEPS,
            n_features=N_FEATURES,
            n_heads=2,
            n_layers=n_layers,
        )

    raise AssertionError(f'Unknown model "{name}"')


def get_summary_writer(*, name: str, epochs: int, lr: float):
    '''
    Creates the summary writer for the given model.
    The parameters are used to distinguish the output events file.
    The events file will be placed in the `runs` directory by default.
    '''
    comment = f'_{name}_e{epochs}_lr{lr:.0e}'
    return SummaryWriter(comment=comment)


if __name__ == '__main__':
    fire.Fire(train_models)
