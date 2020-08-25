from pathlib import Path

from predictor.models import TFAkiLstm, TFAkiGpt2

import fire
import logging
import numpy as np
import tensorflow as tf

# set random seed (for reproducibility)
np.random.seed(7)
tf.random.set_seed(7)

# constants
TIMESTEPS = 8
N_FEATURES = 16


def train_models(
    epochs: int = 1,
    batch_size: int = 256,
    dataset_dir: str = 'dataset',
    ckpt_dir: str = 'saved_models',
    log_dir: str = 'logs',
    training: str = 'matrix_training.npy',
    val: str = 'matrix_validation.npy',
):
    '''
    Trains 3 models (LSTM, GPT-2 and CNN) to predict next-day AKI.

    Parameters:
    epochs: For how many epochs we train the models
    batch_size: The batch size to be used during training (the bigger the better)
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
    # check cuda availability
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        print('CUDA is not available. Training will be slow.')

    # convert dir names to dir paths
    dataset_path = Path(dataset_dir)
    ckpt_path = Path(ckpt_dir)
    log_path = Path(log_dir)

    # verify training and validation data's existence
    train_path = dataset_path / training
    val_path = dataset_path / val
    assert train_path.exists(), f'{training} does not exist'
    assert val_path.exists(), f'{val} does not exist'

    # load training and validation data
    train_matrix = np.load(train_path).astype(np.float32)
    train_x = train_matrix[:, :, :-1]
    train_y = train_matrix[:, :, -1:]
    val_matrix = np.load(val_path).astype(np.float32)
    val_x = val_matrix[:, :, :-1]
    val_y = val_matrix[:, :, -1:]

    # prepare training keyword args
    training_kwargs = {
        'x': train_x,
        'y': train_y,
        'epochs': epochs,
        'batch_size': batch_size,
        'shuffle': True,
        'validation_data': (val_x, val_y),
    }

    # train all models
    train('lstm', training_kwargs, ckpt_path=ckpt_path, log_path=log_path)
    train('gpt2', training_kwargs, ckpt_path=ckpt_path, log_path=log_path)


def train(name: str, training_kwargs, *, ckpt_path: Path, log_path: Path):
    model = get_model(name)
    model.compile(
        optimizer='adam',
        loss=[
            tf.keras.losses.BinaryCrossentropy(from_logits=False),
            None,  # output 2 is attn weights (doesn't need to train)
        ],
        metrics=[
            [
                'acc',
                tf.keras.metrics.AUC(name='auc'),
            ],
            None,  # output 2 is attn weights (doesn't need to measure)
        ],
    )

    # train model with tensorboard callback (for graphing)
    model.fit(
        callbacks=[tf.keras.callbacks.TensorBoard(
            log_dir=log_path / name,
            histogram_freq=1,
        )],
        **training_kwargs,
    )

    # only save model weights
    model_name = f'{name}_e{training_kwargs["epochs"]}'
    model_weights_path = ckpt_path / model_name / name
    model.save_weights(model_weights_path)


def get_model(name: str):
    if name == 'lstm':
        return TFAkiLstm(
            timesteps=TIMESTEPS,
            n_features=N_FEATURES,
        )

    if name == 'gpt2':
        return TFAkiGpt2(
            n_heads=2,
            timesteps=TIMESTEPS,
            n_features=N_FEATURES,
            n_layers=1,
        )

    raise AssertionError(f'Unknown model "{name}"')


if __name__ == '__main__':
    fire.Fire(train_models)
