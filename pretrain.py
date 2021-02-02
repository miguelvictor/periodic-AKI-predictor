from argparse import ArgumentParser
from models import PredictiveModel1
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
import warnings


if __name__ == '__main__':
    # Initialize main args parser
    parser = ArgumentParser()

    # add program-level args
    parser.add_argument('--dataset_dir', type=str,
                        default='pretraining_data.npy')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_workers', type=int, default=8)

    # add model-specific args
    parser = PredictiveModel1.add_model_specific_args(parser)

    # add available trainer options
    parser = Trainer.add_argparse_args(parser)

    # initialize trainer and model
    args = parser.parse_args()
    trainer = Trainer.from_argparse_args(args)
    model = PredictiveModel1(**vars(args))

    # prepare dataset and dataloader
    data = np.load(args.dataset_dir)
    dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    # warn for not using GPUs
    if torch.cuda.is_available() and args.gpus is None:
        warnings.warn('CUDA is available but not used.')

    # start training
    trainer.fit(model, dataloader)
