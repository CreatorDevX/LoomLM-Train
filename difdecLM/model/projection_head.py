import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedLMHead(nn.Module):
    """Fully learned LM head — dense gradient directly to decoder. No frozen embedding bottleneck."""
    def __init__(self, d_decoder, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_decoder, vocab_size)
        self.eos_head = nn.Linear(d_decoder, 1)

    def forward(self, h):
        return self.proj(h), self.eos_head(h).squeeze(-1)


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
        with torch.amp.autocast('cuda', enabled=False):
            h_f32 = h.float() if h.dtype != torch.float32 else h
            logits = F.linear(h_f32, self.proj.weight.float())
            eos_logits = F.linear(h_f32, self.eos_head.weight.float(), self.eos_head.bias.float() if self.eos_head.bias is not None else None).squeeze(-1)
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

        self.lora_A = None  # attached by train.py after backbone.apply_lora()
        self.lora_B = None
        self.lora_scale = 1.0

    def attach_lora_embedding(self, lora_A, lora_B, scale):
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_scale = scale

    def forward(self, h):
        with torch.amp.autocast('cuda', enabled=False):
            h_f32 = h.float() if h.dtype != torch.float32 else h
            h_up = F.linear(h_f32, self.up_proj.weight.float(), self.up_proj.bias.float() if self.up_proj.bias is not None else None)

            # LoRA embedding correction: h_up @ B^T @ A^T * scale
            if self.lora_A is not None:
                h_lora = F.linear(h_up, self.lora_B.float())
                h_lora = F.linear(h_lora, self.lora_A.float())
                h_up = h_up + h_lora * self.lora_scale

            logits = F.linear(h_up, self.embedding_weight.float())
            eos_logits = F.linear(h_f32, self.eos_head.weight.float(), self.eos_head.bias.float() if self.eos_head.bias is not None else None).squeeze(-1)
        return logits, eos_logits
