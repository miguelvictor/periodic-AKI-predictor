from argparse import ArgumentParser
from evaluate import evaluate
from models import PredictiveModel2
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import os
import sys
import torch
import warnings


def parse_args():
    # Initialize main args parser
    parser = ArgumentParser()

    # add model-specific args
    parser = PredictiveModel2.add_model_specific_args(parser)

    # add available trainer options
    parser = Trainer.add_argparse_args(parser)

    # add program-level args
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--training_set', type=str,
                        default='finetuning_training.npy')
    parser.add_argument('--validation_set', type=str,
                        default='finetuning_validation.npy')
    parser.add_argument('--testing_set', type=str,
                        default='finetuning_testing.npy')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_workers', type=int, default=8)

    # initialize trainer and model
    return parser.parse_args()


def create_dataloader(path, batch_size, n_workers):
    data = np.load(path)
    dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))

    print(f'Found {data.shape[0]} samples in {path}')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )


if __name__ == '__main__':
    args = parse_args()

    # lightning_logs directory should be cleared before finetuning
    # because the directory will contain both models (pretrained and not pretrained)
    lightning_logs_path = 'lightning_logs'
    if os.path.isdir(os.path.join(lightning_logs_path, 'version_0')):
        print('Lightning logs directory already contains version0 ckpt.')
        sys.exit(1)

    # initialize trainer and model
    trainer = Trainer.from_argparse_args(args)

    # load pretrained model if path is given
    if args.path:
        print('Will load pretrained weights for finetuning.')
        model = PredictiveModel2.from_pretrained(**vars(args))
    else:
        print('Will train model from scratch.')
        model = PredictiveModel2(**vars(args))

    # prepare dataset and dataloader
    training_dl = create_dataloader(
        args.training_set, args.batch_size, args.n_workers)
    validation_dl = create_dataloader(
        args.validation_set, args.batch_size, args.n_workers)

    # warn for not using GPUs
    if torch.cuda.is_available() and args.gpus is None:
        warnings.warn('CUDA is available but not used.')

    # train and test the model
    trainer.fit(
        model,
        train_dataloader=training_dl,
        val_dataloaders=[validation_dl],
    )

    # show performance on the testing set
    # load model again because the current model may not have the "best" weights
    ckpt_path = os.path.join(lightning_logs_path, 'version_0', 'checkpoints')
    ckpt_fname = os.listdir(ckpt_path)[0]
    evaluate(os.path.join(ckpt_path, ckpt_fname), args.testing_set)
