import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    def __init__(self, d_decoder, d_context):
        super().__init__()
        self.gamma_beta = nn.Linear(d_context, 2 * d_decoder)

    def forward(self, h, context):
        gamma_beta = self.gamma_beta(context)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * h + beta


class HybridConditioning(nn.Module):
    def __init__(self, d_decoder, d_context, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_decoder * 2
        self.net = nn.Sequential(
            nn.Linear(d_context + d_decoder, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, 2 * d_decoder),
        )

    def forward(self, h, context):
        context_expanded = context.unsqueeze(1).expand(-1, h.size(1), -1)
        combined = torch.cat([h, context_expanded], dim=-1)
        gamma_beta = self.net(combined)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * h + beta


class CrossAttention(nn.Module):
    def __init__(self, d_decoder, d_context, n_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_decoder,
            num_heads=n_heads,
            kdim=d_context,
            vdim=d_context,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, context):
        out, _ = self.attn(x, context, context, need_weights=False)
        return out
