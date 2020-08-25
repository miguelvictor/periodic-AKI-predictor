import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf: int, nx: int):
        super().__init__()

        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, timesteps, n_features, n_heads):
        super().__init__()

        assert n_features % n_heads == 0
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((n_features, n_features), dtype=torch.uint8)).view(
                1, 1, n_features, n_features)
        )
        self.register_buffer(
            "masked_bias",
            torch.tensor(-1e4)
        )

        self.timesteps = timesteps
        self.n_features = n_features
        self.n_heads = n_heads

        self.c_attn = Conv1D(n_features * 3, n_features)
        self.c_proj = Conv1D(n_features, n_features)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def _attn(self, q, k, v, attn_mask):
        # q, k, p, v have shape [batch, heads, days, head_features]
        w = torch.matmul(q, k)
        w = w / (float(v.size(-1)) ** 0.5)

        # if only "normal" attention layer implements causal mask
        nd, ns = w.size(-2), w.size(-1)
        causal_mask = self.bias[:, :, ns - nd: ns, :ns]
        w = torch.where(causal_mask.bool(), w, self.masked_bias.to(w.dtype))

        # Apply the attention mask
        w = w + attn_mask

        w = F.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        # return outputs and attention weights
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_heads,
                                       x.size(-1) // self.n_heads)
        x = x.view(*new_x_shape)
        perm = (0, 2, 3, 1) if k else (0, 2, 1, 3)
        return x.permute(*perm)

    def forward(self, x, mask):
        q, k, v = self.c_attn(x).split(self.n_features, dim=2)
        q = self.split_heads(q)
        k = self.split_heads(k, k=True)
        v = self.split_heads(v)

        a = self._attn(q, k, v, mask)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return a


class MLP(nn.Module):
    def __init__(self, n_features):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()

        self.c_fc = Conv1D(n_features * 4, n_features)
        self.c_proj = Conv1D(n_features, n_features * 4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = F.gelu(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, timesteps, n_features, n_heads):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_features)
        self.attn = Attention(timesteps, n_features, n_heads)
        self.ln_2 = nn.LayerNorm(n_features)
        self.mlp = MLP(n_features)

    def forward(self, x, mask):
        a = self.ln_1(x)
        a = self.attn(a, mask)
        x = x + a

        m = self.ln_2(x)
        m = self.mlp(m)
        x = x + m

        return x


class AkiGpt2(nn.Module):
    def __init__(self, *, n_heads, timesteps=8, n_features=16, n_layers=1):
        super().__init__()

        self.n_features = n_features
        self.timesteps = timesteps
        self.n_layers = n_layers

        self.input_ln = nn.LayerNorm(n_features)
        self.attn = nn.Linear(n_features, n_features)
        self.attn_drop = nn.Dropout(0.1)
        self.wpe = nn.Embedding(timesteps, n_features)
        self.h = nn.ModuleList([
            Block(timesteps, n_features, n_heads)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_features)
        self.proj = nn.Linear(n_features, 1)

    def forward(self, x):
        # sanity check
        assert len(x.shape) == 3, \
            f'Input shape should be [batch_size, timesteps, n_features]'
        batch_size, timesteps, n_features = x.shape
        assert timesteps == self.timesteps
        assert n_features == self.n_features

        # create attention mask so that the model won't
        # treat padding days as inputs
        attn_mask = x.bool().any(dim=-1)[:, :, None]
        attn_mask = attn_mask.to(dtype=next(self.parameters()).dtype)
        attn_mask = (1.0 - attn_mask) * -10000.0

        # normalize feature values
        # and apply embedding-level attention
        x = self.input_ln(x)
        w = F.softmax(self.attn(x) + attn_mask, dim=1)
        w = self.attn_drop(w)
        x = x * w

        # add positional encoding
        position_ids = torch.arange(
            timesteps, dtype=torch.long, device=x.device)
        position_ids = position_ids[None, :]
        x = x + self.wpe(position_ids)

        # feed x to n decoder blocks
        attn_mask = attn_mask.view(batch_size, -1)[:, None, None, :]
        for block in self.h:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        x = torch.sigmoid(self.proj(x))

        return x, w
