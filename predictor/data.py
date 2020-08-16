from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torch


class Mimic3Dataset(Dataset):
    def __init__(self, path, pad=True, timesteps=8, transform=None):
        self.pad = pad
        self.timesteps = timesteps
        self.transform = transform
        self.columns = None  # will be set by init_data
        self.data = list(self.init_data(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        x = sample[:, :-1]
        y = sample[:, -1:]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def init_data(self, path):
        df = pd.read_csv(path)
        df.columns = map(str.lower, df.columns)

        unneeded_columns = ['icu_day', 'icustay_id', 'subject_id', 'hadm_id']
        self.columns = [c for c in df.columns if c not in unneeded_columns]

        for _, group in df.groupby('icustay_id'):
            group = group.loc[:, self.columns].values
            yield group if not self.pad else self.__add_padding(group)

    def __add_padding(self, group):
        n_rows = self.timesteps - group.shape[0]
        n_cols = len(self.columns)
        padding = np.zeros((n_rows, n_cols))
        return np.vstack([group, padding])
