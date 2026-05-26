# LoomLM Architecture

## Overview

LoomLM = frozen SmolLM2-135M backbone + trainable diffusion decoder head.
Input: token sequence → backbone hidden states → noisy block embeddings → denoised embeddings → logits.

```
tokens ──► SmolLM2Backbone ──► backbone_hidden [B, S, 576]
                                   │
                                   ├──► prepare_context ──► context [B*n_blk, 576]
                                   │
                                   └──► embed_tokens ──► E0 [B, n_blk, 64, 384]
                                                            │
                              noise ◄── randn_like(E0)       │
                                │                            │
                                ▼                            ▼
                          E_t = √ᾱ·E0 + √(1-ᾱ)·noise  [B*n_blk, 64, 384]
                                │
                                ▼
                    ┌───────────────────────┐
                    │  DiffusionDecoderStack │◄── context + time_emb
                    │  (6 layers, 384d)      │
                    └───────────────────────┘
                                │
                    pred = ε_pred [B*n_blk, 64, 384]
                                │
                                ▼
                    ┌───────────────────────┐
                    │ EmbeddingProjectionHead│
                    └───────────────────────┘
                        │            │
                    logits       eos_logits
                   [B,16,64,49152]  [B,16,64]
```

---

## Component Details

### 1. SmolLM2Backbone (`model/backbone.py`)
- Loads `HuggingFaceTB/SmolLM2-135M` (Llama-based, 576 hidden dim)
- Strips LM head (`self.lm_head = None`)
- Exposes `embed_fn` (embedding table, vocab=49152), `layers` (30 LlamaDecoderLayers), `norm` (final RMSNorm)
- `forward(input_ids, attention_mask)` → calls `self.backbone(input_ids, attention_mask)` → returns `last_hidden_state [B, S, 576]`
- Registers `embedding_weight` buffer (clone of embed table, used by projection head)
- Supports `freeze` / `unfreeze_last_n_layers` / `apply_lora(r, alpha)`

### 2. LoRA Adapters (`model/backbone.py`)
- `LoRALinear`: wraps any `nn.Linear` with low-rank adapters A:[in,r], B:[r,out]
  - Forward: `original(x) + (x @ A @ B) * (alpha/r)`
  - Original is frozen; A (Kaiming init) and B (zero init) are trainable
- `LoRAEmbedding`: same for `nn.Embedding`
  - Forward: `original(x) + (h @ A @ B) * (alpha/r)` where h = original(x)
- `apply_lora(r, alpha)` on `SmolLM2Backbone`: recurses through `self.backbone` children, wraps all `nn.Linear` + embedding

### 3. TimeEmbedding (`model/time_embedding.py`)
- Sinusoidal position encoding (half_dim = d_model//2)
- Optional MLP: Linear(d, 2d) → SiLU → Linear(2d, d)
- d_model = 384

### 4. FiLMLayer (`model/conditioning.py`)
- `Linear(d_context, 2*d_decoder)` → chunk → gamma, beta
- Applied per-layer: `gamma * h + beta` where gamma/beta broadcast over sequence dim

### 5. DiffusionDecoderLayer (`model/diffusion_decoder.py`)
```
x ──► LayerNorm ──► MHA ──► +x ──► * (1 + time_gate) ──► FiLM ──► LayerNorm ──► FFN ──► +x
```
- MHA: `nn.MultiheadAttention(384, 6 heads, batch_first)`
- Time gate: `Linear(384, 384)` → sigmoid? No, just `(1 + linear(time_emb))` elementwise multiply
- FFN: Linear(384, 1536) → SiLU → Dropout → Linear(1536, 384) → Dropout
- No cross-attention; backbone context injected via FiLM only

### 6. DiffusionDecoderStack (`model/diffusion_decoder.py`)
- `context_proj`: Linear(576, 384) — projects backbone context into decoder dim
- 6 stacked `DiffusionDecoderLayer` layers
- Final LayerNorm

### 7. EmbeddingProjectionHead (`model/projection_head.py`)
- `up_proj`: Linear(384, 576) — projects decoder output back to backbone dim
- `logits = F.linear(h_up, embedding_weight)` — dot product with frozen SmolLM2 embedding table → vocab logits
- `eos_head`: Linear(384, 1) → squeeze → per-position EOS score
- Total: ~14.6M params (no separate decoder embedding; shares backbone's)

### 8. DifDecLM Assembly (`model/difdec_lm.py`)
- `prepare_context(backbone_hidden, n_blocks)`: mean-pool last `context_window` hidden positions per block
- `get_clean_embeddings(block_tokens)`: embed_tokens → embed_proj (shared mode)
- `predict_noise(noise_pred, E_t, sqrt_α, sqrt_1-α)`: converts prediction type (ε/x₀/v)
- Cached cosine schedule for `sqrt_ᾱ`
- Forward returns dict with keys:
  - `noise_pred`, `noise_target`, `pred_embeddings`, `clean_embeddings`, `noisy_embeddings`, `logits`, `eos_logits`, `context`

### 9. Diffusion Process (`training/diffusion_process.py`)
- Cosine noise schedule: `ᾱ(t) = cos²((t/T + s)/(1+s) · π/2)`, clamped [0,1]
- `get_timesteps(B, n_blocks)`: random uniform timesteps [0, T) per block
- DDIM sampling: reverse from t=T-1 to t=0 in K steps
  - Predict ε, compute x₀, step to next timestep
  - `predict_noise` converts ε/x₀/v prediction as needed

### 10. Losses (`training/losses.py`)
All inputs cast to `float32` internally to prevent fp16 overflow.

#### Diffusion Loss (ε-prediction MSE)
```python
bm = block_mask.unsqueeze(-1).unsqueeze(-1).float()  # [B, n_blk, 1, 1]
diff_loss = MSE(noise_pred * bm, noise_target * bm, reduction="sum")
diff_loss /= bm.sum().clamp(min=1)  # mean over real (non-padded) blocks
```
- Inputs: `noise_pred` [B, n_blk, 64, 384] (decoder output), `noise_target` (randn noise)
- Block-masked: padded blocks (where block_mask=False) contribute 0 to sum
- No block_mask → `reduction="mean"` over all elements

#### Token CLM Loss (Cross-Entropy)
```python
token_loss = CE(logits.view(-1, 49152), block_tokens.view(-1), reduction="none")
             .view(B, n_blk, 64)
# If mask_pad: multiply by pad_mask * block_mask, then sum / count
```
- Weighted by `clm_weight` (ramps 0→max over `clm_ramp_steps`)
- Pad-masked: positions where `block_tokens == 0` (pad_id) are excluded
- Block-masked: entire padded blocks are excluded

#### EOS Loss (Binary Cross-Entropy)
```python
eos_target[:, :, -1] = 1.0  # only last position of each block is EOS
eos_loss = BCE_with_logits(eos_logits, eos_target, reduction="none")
eos_loss *= block_mask.unsqueeze(-1)
eos_loss = eos_loss.sum() / block_mask.sum()
```
- Block-masked: padded blocks excluded from the mean
- No block_mask → `reduction="mean"`

#### Total Loss
```python
total = diff_weight * diff_loss + clm_weight * token_loss + eos_weight * eos_loss
```

---

## Phase Presets

| | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| **Steps** | 1500 | 500 | 3000 |
| **LR** | 1e-4 | 2e-5 | 2e-5 |
| **Backbone** | frozen | frozen + LoRA r=16 | unfreeze last 4 layers |
| **Diff weight** | 1.0 | 1.0 | 0.5 |
| **CLM weight** | 0.0 | 0.3 (ramp 3k) | 0.5 (ramp 2k) |
| **EOS weight** | 0.01 | 0.05 | 0.02 |
| **Sampling steps** | 4 | 4 | 4 |
| **Trainable** | ~14.6M (decoder head) | ~14.6M + LoRA (~4.5M) | ~14.6M + 4 backbone layers (~18M) |

---

## Data Pipeline (`data/dataset.py`)

```
HuggingFaceFW/fineweb-edu (streaming, 50k samples)
    │
    ▼
BlockDiffusionDataset (IterableDataset)
    ├── tokenizer.encode(text, max_length=1024)
    ├── pad to block_size+1 if short
    ├── truncate to nearest block boundary (input_ids[:-1], block_tokens[1:])
    ├── reshape block_tokens to [n_blocks, 64]
    └── yield {input_ids, attention_mask, block_tokens}
    │
    ▼
FixedLengthDataset (wraps above)
    ├── pad/crop every sample to exactly 1024 tokens (16 blocks)
    └─► torch.compile-friendly fixed shapes
    │
    ▼
DataLoader(batch_size=16, collate_fn=collate_blocks, pin_memory=True)
    │
    ▼
collate_blocks: pad to max in batch → {input_ids, attention_mask, block_tokens, block_counts, block_mask}
```

With fixed-length sequences, `block_mask` is all `True` (no padding within batch), but the loss functions still check it.

---

## Parameter Counts

| Component | Params |
|---|---|
| Backbone (SmolLM2-135M, frozen) | 134,515,008 |
| TimeEmbedding (384→768→384) | 442,368 |
| DiffusionDecoderStack (6 layers) | 11,881,728 |
| EmbeddingProjectionHead (up_proj + eos_head) | 221,761 |
| embed_proj (Linear 576→384) | 221,185 |
| **Total trainable (phase 1)** | **~14,564,929** |
| **Total model** | **~149,079,937** |

---

## Training Loop (`train.py`)

```
while step < max_steps:
    optimizer.zero_grad()
    for micro in range(grad_accum=2):
        batch ← DataLoader
        timesteps = randint(0, 1000, [B, 16])
        with autocast(fp16):
            outputs = model(input_ids, timesteps, attention_mask)
            loss = loss_fn(outputs, block_tokens, block_mask)
            loss = loss / grad_accum
        loss.backward()
    clip_grad_norm_(1.0)
    optimizer.step()
    scheduler.step()  # cosine, 100-step warmup
    step += 1
    # log, eval, inference, save every N steps
```
