from argparse import ArgumentParser
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, Block
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super(BaseModel, self).__init__()
        self.config = config
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            Block(config.n_ctx, config, scale=True)
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # input data's should be a 3D tensor
        assert x.ndim == 3
        _, n_days, n_features = x.size()
        assert n_days <= self.config.n_positions and n_features == self.config.n_embd

        # add position embeddings
        position_ids = torch.arange(n_days, dtype=torch.long, device=x.device)
        hidden_states = x + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        # go through layer blocks
        for block in self.h:
            outputs = block(
                hidden_states,
                layer_past=None,
                attention_mask=attn_mask,
                head_mask=None,  # TODO: maybe add this ?
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=True,
                output_attentions=False,
            )
            hidden_states = outputs[0]

        # go through final layer normalization layer
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class PredictiveModel1(pl.LightningModule):
    def __init__(
        self,
        n_days=8,
        n_features=16,
        n_layers=16,
        n_head=4,
        d_model=1024,
        learning_rate=1e-3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            'n_days', 'n_features', 'n_layers', 'n_head', 'd_model', 'learning_rate')
        config = self.create_config()
        self.base = BaseModel(config)
        self.head = nn.Linear(n_features, n_features)
        self.drop = nn.Dropout(config.resid_pdrop)

    def create_config(self):
        assert self.hparams.n_features % self.hparams.n_head == 0
        return GPT2Config(
            n_positions=self.hparams.n_days,
            n_embd=self.hparams.n_features,
            n_ctx=self.hparams.d_model,
            n_inner=self.hparams.d_model,
            n_head=self.hparams.n_head,
            n_layer=self.hparams.n_layers,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # retrieve batch size from input tensor
        assert x.ndim == 3
        batch_size, timesteps, _ = x.shape

        # retrieve by going through GPT-2 body
        hidden_states = self.base(x, attn_mask=attn_mask)

        # project hidden states
        output_values = self.head(hidden_states)
        output_values = self.drop(output_values)

        # also apply attention mask for the head layer
        if attn_mask is not None:
            attn_mask = (~attn_mask.bool()).float()
            attn_mask = attn_mask.view(batch_size, timesteps, 1)
            output_values = output_values * attn_mask

        return output_values

    def training_step(self, batch, _):
        # unpack list if using TensorDataset
        batch = batch[0] if isinstance(batch, list) else batch
        assert batch.ndim == 3 and batch.size(1) > 1

        # shift input for inputs and targets
        x, y = batch[:, :-1], batch[:, 1:]

        # create attention mask to exclude padded days
        # note that the attention mask should be based on the labels y
        # since inputs x may contain data without the necessary label y
        attn_mask = y.bool().any(dim=-1).float()
        attn_mask = (1 - attn_mask) * -10000.0
        attn_mask = attn_mask[:, None, None, :]

        # do forward pass and compute loss
        y_hat = self(x, attn_mask=attn_mask)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_days', type=int, default=8)
        parser.add_argument('--n_features', type=int, default=16)
        parser.add_argument('--n_layers', type=int, default=16)
        parser.add_argument('--n_head', type=int, default=4)
        parser.add_argument('--d_model', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser


class PredictiveModel2(pl.LightningModule):
    def __init__(
        self,
        n_days=8,
        n_features=16,
        n_layers=16,
        n_head=4,
        d_model=1024,
        learning_rate=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            'n_days', 'n_features', 'n_layers', 'n_head', 'd_model', 'learning_rate')
        config = self.create_config()
        self.base = BaseModel(config)
        self.head = nn.Linear(config.n_embd, 1)
        self.drop = nn.Dropout(config.resid_pdrop)

    def create_config(self):
        assert self.hparams.n_features % self.hparams.n_head == 0
        return GPT2Config(
            n_positions=self.hparams.n_days,
            n_embd=self.hparams.n_features,
            n_ctx=self.hparams.d_model,
            n_inner=self.hparams.d_model,
            n_head=self.hparams.n_head,
            n_layer=self.hparams.n_layers,
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # retrieve batch size from input tensor
        assert x.ndim == 3
        batch_size, timesteps, _ = x.shape

        # retrieve by going through GPT-2 body
        hidden_states = self.base(x, attn_mask=attn_mask)

        # project hidden states
        probabilities = self.head(hidden_states)
        probabilities = self.drop(probabilities)
        probabilities = probabilities.view(batch_size, timesteps)

        # also apply attention mask for the head layer
        if attn_mask is not None:
            attn_mask = (~attn_mask.bool()).float()
            attn_mask = attn_mask.view(batch_size, timesteps)
            probabilities = probabilities * attn_mask

        return probabilities

    def training_step(self, batch, _):
        loss = self.__compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        loss = self.__compute_loss(batch)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def __compute_loss(self, batch):
        # unpack list if using TensorDataset
        batch = batch[0] if isinstance(batch, list) else batch
        assert batch.ndim == 3 and batch.size(1) > 1

        # shift input for inputs and targets
        x, y = batch[:, :, :-1], batch[:, :, -1]

        # create attention mask to exclude padded days
        # note that the attention mask should be based on the labels y
        # since inputs x may contain data without the necessary label y
        attn_mask = x.bool().any(dim=-1).float()
        attn_mask = (1 - attn_mask) * -10000.0
        attn_mask = attn_mask[:, None, None, :]

        y_hat = self(x, attn_mask=attn_mask)
        return F.binary_cross_entropy_with_logits(y_hat, y)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_days', type=int, default=8)
        parser.add_argument('--n_features', type=int, default=16)
        parser.add_argument('--n_layers', type=int, default=16)
        parser.add_argument('--n_head', type=int, default=4)
        parser.add_argument('--d_model', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        return parser
