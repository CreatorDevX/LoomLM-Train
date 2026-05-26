import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenProjectionHead(nn.Module):
    def __init__(self, d_decoder, vocab_size, use_weight_tying=True, embedding_weight=None):
        super().__init__()
        self.d_decoder = d_decoder
        self.vocab_size = vocab_size
        self.use_weight_tying = use_weight_tying

        if use_weight_tying and embedding_weight is not None:
            self.proj = nn.Linear(d_decoder, vocab_size, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d_decoder, vocab_size)

        self.eos_head = nn.Linear(d_decoder, 1)

    def forward(self, h):
        with torch.cuda.amp.autocast(enabled=False):
            h_f32 = h.float() if h.dtype != torch.float32 else h
            logits = self.proj(h_f32)
            eos_logits = self.eos_head(h_f32).squeeze(-1)
        return logits, eos_logits


class EmbeddingProjectionHead(nn.Module):
    def __init__(self, d_decoder, d_backbone, vocab_size, embedding_weight):
        super().__init__()
        self.d_decoder = d_decoder
        self.d_backbone = d_backbone
        self.vocab_size = vocab_size

        self.up_proj = nn.Linear(d_decoder, d_backbone)
        self.embedding_weight = embedding_weight
        self.eos_head = nn.Linear(d_decoder, 1)

    def forward(self, h):
        with torch.cuda.amp.autocast(enabled=False):
            h_f32 = h.float() if h.dtype != torch.float32 else h
            h_up = self.up_proj(h_f32)
            logits = F.linear(h_up, self.embedding_weight.float())
            eos_logits = self.eos_head(h_f32).squeeze(-1)
        return logits, eos_logits
