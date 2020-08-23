from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader

import torch


@dataclass
class TrainingArgs:
    epochs: int
    lr: float
    train_dl: DataLoader
    val_dl: DataLoader
    device: torch.device
    ckpt_dir: Path
    n_layers: int = 1
