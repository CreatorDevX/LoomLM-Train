import torch
import torch.nn as nn
import torch.nn.functional as F
from .conditioning import FiLMLayer


class DiffusionDecoderLayer(nn.Module):
    def __init__(self, d_decoder, n_heads, d_ff, dropout, d_context, d_time):
        super().__init__()
        self.d_decoder = d_decoder
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_decoder)
        self.self_attn = nn.MultiheadAttention(
            d_decoder, n_heads, dropout=dropout, batch_first=True
        )

        self.time_mlp = nn.Linear(d_time, d_decoder)

        self.film = FiLMLayer(d_decoder, d_context)

        self.norm2 = nn.LayerNorm(d_decoder)

        self.ffn = nn.Sequential(
            nn.Linear(d_decoder, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_decoder),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, time_emb, attn_mask=None):
        x = x + self._sa_block(self.norm1(x), attn_mask)
        time_gate = self.time_mlp(time_emb).unsqueeze(1)
        x = x * (1 + time_gate)
        x = self.film(x, context)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x, attn_mask=None):
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return self.dropout(x)

    def _ff_block(self, x):
        return self.ffn(x)


class DiffusionDecoderStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        dc = config.decoder
        difc = config.diffusion
        bc = config.backbone

        self.context_proj = nn.Linear(bc.d_backbone, dc.d_decoder)

        self.layers = nn.ModuleList([
            DiffusionDecoderLayer(
                d_decoder=dc.d_decoder,
                n_heads=dc.n_heads,
                d_ff=dc.d_ff,
                dropout=dc.dropout,
                d_context=dc.d_decoder,
                d_time=difc.d_time_embed,
            )
            for _ in range(dc.n_layers)
        ])

        self.norm = nn.LayerNorm(dc.d_decoder)

    def forward(self, block_emb, context, time_emb):
        context = self.context_proj(context)
        h = block_emb
        for layer in self.layers:
            h = layer(h, context, time_emb)
        h = self.norm(h)
        return h
