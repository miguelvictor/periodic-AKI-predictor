from models import PredictiveModel1
from torch.utils.data import TensorDataset, DataLoader

import fire
import numpy as np
import torch
import pytorch_lightning as pl


def pretrain(epochs: int = 100, batch_size: int = 8):
    data = np.load('dataset/pretraining_data.npy')
    dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    model = PredictiveModel1()

    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    fire.Fire(pretrain)
