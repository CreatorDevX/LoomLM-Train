import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    def __init__(self, d_model, use_mlp=True):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer("freqs", torch.exp(-emb * torch.arange(half_dim)))

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model),
            )
        else:
            self.mlp = None

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t.unsqueeze(-1).float() * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.d_model % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        if self.mlp is not None:
            emb = self.mlp(emb)

        return emb
