import torch
import torch.nn as nn


class AkiLstm(nn.Module):
    def __init__(self, n_features=16, timesteps=8, n_layers=1, hidden_dim=256):
        super(AkiLstm, self).__init__()

        self.n_features = n_features
        self.timesteps = timesteps

        self.norm = nn.LayerNorm(
            normalized_shape=(timesteps, n_features),
            elementwise_affine=True,
        )
        self.attn = nn.Linear(
            in_features=n_features,
            out_features=n_features,
        )
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(
            in_features=hidden_dim,
            out_features=1,
        )

    def forward(self, x):
        # input sanity check
        assert len(x.shape) == 3, 'Input of the model should be a 3D Tensor'
        _, timesteps, n_features = x.shape
        assert self.timesteps == timesteps, 'Input timesteps axis is invalid'
        assert self.n_features == n_features, 'Input n_features axis is invalid'

        # apply per-element normalization
        x = self.norm(x)

        # apply feature-level attention
        w = self.attn(x)
        x = x * w

        # compute los for each of the samples
        # and pack the sequence for LSTM
        x_lengths = x.bool().any(dim=-1).sum(dim=-1)
        x = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True, enforce_sorted=False)

        # go through LSTM and unpack results
        x, _ = self.lstm(x)  # TODO: reinit hidden states every after epoch
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # project results into a prediction
        x = self.proj(x)
        x = torch.sigmoid(x)

        return x, w
