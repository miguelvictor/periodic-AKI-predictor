from transformers.models.gpt2.modeling_gpt2 import GPT2Config, Block
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TIMESTEPS = 8
N_FEATURES = 16
CONFIG = GPT2Config(
    n_embd=N_FEATURES, n_ctx=1024, n_inner=1024,
    n_head=4, n_layer=16,
)
assert N_FEATURES % CONFIG.n_head == 0


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.wpe = nn.Embedding(N_TIMESTEPS, N_FEATURES)
        self.drop = nn.Dropout(CONFIG.embd_pdrop)
        self.h = nn.ModuleList([
            Block(CONFIG.n_ctx, CONFIG, scale=True)
            for _ in range(CONFIG.n_layer)
        ])
        self.ln_f = nn.LayerNorm(N_FEATURES, eps=CONFIG.layer_norm_epsilon)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # input data's should be a 3D tensor
        assert x.ndim == 3
        _, n_timesteps, n_features = x.size()
        assert n_timesteps <= N_TIMESTEPS and n_features == N_FEATURES

        # add position embeddings
        position_ids = torch.arange(
            n_timesteps, dtype=torch.long, device=x.device)
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
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()

        # initialize model hyperparameters
        self.lr = learning_rate

        # initialize model layers
        self.base = BaseModel()
        self.head = nn.Linear(N_FEATURES, N_FEATURES)
        self.drop = nn.Dropout(CONFIG.resid_pdrop)

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PredictiveModel2(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-5):
        super().__init__()

        # initialize model hyperparameters
        self.lr = learning_rate

        # initialize model layers
        self.base = BaseModel()
        self.head = nn.Linear(N_FEATURES, 1)
        self.drop = nn.Dropout(CONFIG.resid_pdrop)

    def forward(self, x: torch.Tensor):
        hidden_states = self.base(x)
        probabilities = self.head(hidden_states)
        probabilities = self.drop(probabilities)
        return probabilities

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
