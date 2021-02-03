from argparse import ArgumentParser
from models import PredictiveModel2
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
import warnings


def parse_args():
    # Initialize main args parser
    parser = ArgumentParser()

    # add program-level args
    parser.add_argument('--training_set', type=str,
                        default='finetuning_training.npy')
    parser.add_argument('--validation_set', type=str,
                        default='finetuning_validation.npy')
    parser.add_argument('--testing_set', type=str,
                        default='finetuning_testing.npy')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_workers', type=int, default=8)

    # add model-specific args
    parser = PredictiveModel2.add_model_specific_args(parser)

    # add available trainer options
    parser = Trainer.add_argparse_args(parser)

    # initialize trainer and model
    return parser.parse_args()


def create_dataloader(path, batch_size, n_workers):
    data = np.load(path)
    dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )


if __name__ == '__main__':
    args = parse_args()

    # initialize trainer and model
    trainer = Trainer.from_argparse_args(args)
    model = PredictiveModel2(**vars(args))

    # prepare dataset and dataloader
    training_dl = create_dataloader(
        args.training_set, args.batch_size, args.n_workers)
    validation_dl = create_dataloader(
        args.validation_set, args.batch_size, args.n_workers)
    testing_dl = create_dataloader(
        args.testing_set, args.batch_size, args.n_workers)

    # warn for not using GPUs
    if torch.cuda.is_available() and args.gpus is None:
        warnings.warn('CUDA is available but not used.')

    # train and test the model
    trainer.fit(
        model,
        train_dataloader=training_dl,
        val_dataloaders=[validation_dl],
    )
    trainer.test(model, test_dataloaders=[testing_dl])
