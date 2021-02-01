from models import PredictiveModel1
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional

import fire
import numpy as np
import torch
import pytorch_lightning as pl


def pretrain(
    dataset_dir: str = 'pretraining_data.npy',
    learning_rate: float = 1e-5,
    epochs: int = 500,
    batch_size: int = 512,
    gpus: Optional[int] = None,
):
    # prepare dataset and dataloader
    data = np.load(dataset_dir)
    dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    # instantiate model for training
    model = PredictiveModel1(learning_rate=learning_rate)

    # train model
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus,
        num_nodes=1,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    fire.Fire(pretrain)
