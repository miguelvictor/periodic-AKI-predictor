from argparse import ArgumentParser
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, Block
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# default hyperparameters shared by both models
N_TIMESTEPS = 8
N_FEATURES = 16
N_LAYERS = 16
N_HEAD = 4
D_MODEL = 1024
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-5
assert N_FEATURES % N_HEAD == 0


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
        n_days=N_TIMESTEPS,
        n_features=N_FEATURES,
        n_layers=N_LAYERS,
        n_head=N_HEAD,
        d_model=D_MODEL,
        learning_rate=PRETRAIN_LR,
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
        parser.add_argument('--n_days', type=int, default=N_TIMESTEPS)
        parser.add_argument('--n_features', type=int, default=N_FEATURES)
        parser.add_argument('--n_layers', type=int, default=N_LAYERS)
        parser.add_argument('--n_head', type=int, default=N_HEAD)
        parser.add_argument('--d_model', type=int, default=D_MODEL)
        parser.add_argument('--learning_rate', type=float, default=PRETRAIN_LR)
        return parser


class PredictiveModel2(pl.LightningModule):
    def __init__(
        self,
        n_days=N_TIMESTEPS,
        n_features=N_FEATURES,
        n_layers=N_LAYERS,
        n_head=N_HEAD,
        d_model=D_MODEL,
        learning_rate=FINETUNE_LR,
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
        parser.add_argument('--n_days', type=int, default=N_TIMESTEPS)
        parser.add_argument('--n_features', type=int, default=N_FEATURES)
        parser.add_argument('--n_layers', type=int, default=N_LAYERS)
        parser.add_argument('--n_head', type=int, default=N_HEAD)
        parser.add_argument('--d_model', type=int, default=D_MODEL)
        parser.add_argument('--learning_rate', type=float, default=FINETUNE_LR)
        return parser

    @staticmethod
    def from_pretrained(path, **kwargs):
        pretrained = PredictiveModel1.load_from_checkpoint(path)

        # hyperparameters for the base model should be the same for both models
        hparams = pretrained.hparams
        assert hparams.n_days == kwargs.get('n_days', N_TIMESTEPS)
        assert hparams.n_features == kwargs.get('n_features', N_FEATURES)
        assert hparams.n_layers == kwargs.get('n_layers', N_LAYERS)
        assert hparams.n_head == kwargs.get('n_head', N_HEAD)
        assert hparams.d_model == kwargs.get('d_model', D_MODEL)

        # remove head layer weights from state dict
        # since the head layer is only used for pretraining and not finetuning
        state_dict = {
            k: v
            for k, v in pretrained.state_dict().items()
            if not k.startswith('head')
        }

        # initialize finetuning model
        model = PredictiveModel2(**kwargs)

        # add the weights for the head layer of the finetuning model
        # the initial weights of the newly initialized model can be used
        state_dict['head.weight'] = model.head.weight
        state_dict['head.bias'] = model.head.bias

        # load all weights from the pretrained model (except the head layer)
        model.load_state_dict(state_dict=state_dict)
        return model
