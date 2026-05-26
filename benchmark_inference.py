"""
Benchmark: SmolLM2-135M AR generation vs DifDecLM diffusion generation.
Compares latency for generating 64 tokens.
"""
import sys; sys.path.insert(0, "difdecLM")
import os
import time
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"  Device: {device} ({gpu_name})")
print(f"  Torch version: {torch.__version__}")

def timed(fn, n_warmup=2, n_trials=5, desc=""):
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    mean_s = float(np.mean(times))
    std_s = float(np.std(times))
    print(f"  {desc}: {mean_s*1000:.1f} +/- {std_s*1000:.1f} ms  (n={n_trials})")
    return mean_s, std_s


# 1. Load SmolLM2-135M

print("\n" + "="*60)
print("  Loading SmolLM2-135M backbone...")
print("="*60)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

backbone_lm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
).to(device)
backbone_lm.eval()

backbone_params = sum(p.numel() for p in backbone_lm.parameters())
print(f"  Backbone params: {backbone_params:,}")


# 2. Build DifDecLM head (random init)

print("\n" + "="*60)
print("  Building DifDecLM diffusion decoder head...")
print("="*60)

from difdecLM.config import DifDecConfig
from difdecLM.model.time_embedding import TimeEmbedding
from difdecLM.model.diffusion_decoder import DiffusionDecoderStack

config = DifDecConfig()
d_back  = config.backbone.d_backbone
d_dec   = config.decoder.d_decoder
V       = config.vocab_size
B       = 1
BLOCK   = config.block.block_size
K       = config.diffusion.sampling_steps

embed_proj  = nn.Linear(d_back, d_dec).to(device)
time_emb    = TimeEmbedding(config.diffusion.d_time_embed, use_mlp=True).to(device)
decoder     = DiffusionDecoderStack(config).to(device)
head_params = sum(p.numel() for p in decoder.parameters()) \
            + sum(p.numel() for p in time_emb.parameters()) \
            + sum(p.numel() for p in embed_proj.parameters())
print(f"  Diffusion head params (decoder+time+proj): {head_params:,}")
print(f"  Total model params (backbone + head): {backbone_params + head_params:,}")
print(f"  K (diffusion steps): {K}")

# Precompute cosine schedule
T = config.diffusion.timesteps
s = config.diffusion.cosine_s
t_lin = torch.arange(T + 1, device=device).float() / T
f_cos = torch.cos((t_lin + s) / (1.0 + s) * (torch.pi / 2.0))
alpha_bar = f_cos.clamp(min=0.0, max=1.0)

# Cache the backbone base model forward for DifDec (no LM head)
# We just use backbone_lm.model (the Llama base model)
base_model = backbone_lm.model
emb_weight = backbone_lm.get_input_embeddings().weight  # [V, 576]

def get_noise_pred(noise_pred_raw, E_t, sqrt_ab, sqrt_1ab):
    ptype = config.diffusion.prediction_type
    if ptype == "epsilon":
        return noise_pred_raw
    elif ptype == "x0":
        return (E_t - sqrt_ab * noise_pred_raw) / sqrt_1ab.clamp(min=1e-8)
    elif ptype == "v":
        return sqrt_ab * noise_pred_raw + sqrt_1ab * E_t
    return noise_pred_raw


# 3. Benchmark AR generation (64 tokens) - NAIVE (full forward each step)

print("\n" + "-"*60)
print("  BENCHMARK 1: Autoregressive - 64 tokens (naive, full forward)")
print("-"*60)

prompt_text = "The future of artificial intelligence will transform"
prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
prompt_len = prompt_ids.shape[1]

def ar_generate_64_naive():
    with torch.no_grad():
        ids = prompt_ids.clone()
        for _ in range(64):
            logits = backbone_lm(ids).logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return ids

ar_naive, ar_naive_std = timed(ar_generate_64_naive, n_warmup=2, n_trials=5, desc="AR 64 tokens (naive full fwd)")


# 4. Benchmark AR generation (64 tokens) - OPTIMIZED (with KV cache)

print("\n" + "-"*60)
print("  BENCHMARK 2: Autoregressive - 64 tokens (HF generate, KV cache)")
print("-"*60)

def ar_generate_64_kv():
    with torch.no_grad():
        out = backbone_lm.generate(
            prompt_ids,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return out

ar_kv, ar_kv_std = timed(ar_generate_64_kv, n_warmup=2, n_trials=5, desc="AR 64 tokens (HF generate, KV cache)")

ar_output_kv = ar_generate_64_kv()
ar_text_kv = tokenizer.decode(ar_output_kv[0, prompt_len:], skip_special_tokens=True)
print(f"  Example output: {ar_text_kv[:120]}...")


# 5. Benchmark AR single-step latency (to extrapolate optimal)

print("\n" + "-"*60)
print("  BENCHMARK 3: AR single step latency (KV cache, 1 new token)")
print("-"*60)

def ar_single_step():
    with torch.no_grad():
        # Use first step with full prompt, then single-token steps
        past = None
        ids = prompt_ids
        for _ in range(1):  # just one step to measure
            out = backbone_lm(input_ids=ids, use_cache=True, past_key_values=past)
            past = out.past_key_values
            next_id = out.logits[:, -1:, :].argmax(dim=-1)
            ids = next_id

ar_step, ar_step_std = timed(ar_single_step, n_warmup=2, n_trials=5, desc="AR single step (KV cache)")

# Extrapolate: 1 prompt encoding + 64 single steps
# Prompt encoding is the same as the first step of naive (full sequence)
ar_opt_est = ar_kv  # already measured
print(f"  (Prompt encoding bundled in HF generate measurement)")


# 6. Benchmark DifDecLM diffusion (64-token block, K=8)

print("\n" + "-"*60)
print(f"  BENCHMARK 4: DifDecLM diffusion - 64 tokens, K={K} steps")
print("-"*60)

def difdec_generate_64():
    with torch.no_grad():
        # 6a. Single backbone forward pass for context
        hidden = base_model(prompt_ids).last_hidden_state  # [1, n, 576]
        ctx = hidden[:, -1:, :].squeeze(1)                 # [1, 576]

        # 6b. DDIM sampling (K steps)
        x_t = torch.randn(B, BLOCK, d_dec, device=device)
        steps = torch.linspace(T - 1, 0, K, device=device).long()

        for i in range(K):
            t_val = int(steps[i].item())
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.float)
            t_emb_val = time_emb(t_batch)

            noise_raw = decoder(x_t, ctx, t_emb_val)      # [1, 64, 384]

            ab_t = alpha_bar[t_val]
            sqrt_ab_t = ab_t.sqrt()
            sqrt_1ab_t = (1.0 - ab_t).clamp(min=0.0).sqrt()

            noise_pred = get_noise_pred(noise_raw, x_t, sqrt_ab_t, sqrt_1ab_t)
            x0_pred = (x_t - sqrt_1ab_t * noise_pred) / sqrt_ab_t.clamp(min=1e-8)

            if i < K - 1:
                ab_next = alpha_bar[int(steps[i+1].item())]
            else:
                ab_next = torch.tensor(1.0, device=device)

            x_t = ab_next.sqrt() * x0_pred + (1.0 - ab_next).clamp(min=0.0).sqrt() * noise_pred

        # 6c. Decode to tokens via shared embedding
        h_up = x_t @ embed_proj.weight     # [1, 64, 576]
        logits = h_up @ emb_weight.T       # [1, 64, V]
        tokens = logits.argmax(dim=-1)
    return tokens

dif_gen, dif_gen_std = timed(difdec_generate_64, n_warmup=2, n_trials=5, desc=f"DifDec {K}x64 tokens")

dif_output = difdec_generate_64()
dif_text = tokenizer.decode(dif_output[0], skip_special_tokens=True)
print(f"  Example output: {dif_text[:120]}...")


# 7. Summary

print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
ar_naive_tps = 64.0 / ar_naive
ar_kv_tps    = 64.0 / ar_kv
dif_tps      = 64.0 / dif_gen
print(f"  {'Method':<35} {'Latency':<15} {'Tokens/s':<12}")
print(f"  {'-'*35} {'-'*15} {'-'*12}")
print(f"  {'AR 64 (naive full fwd)':<35} {ar_naive*1000:<8.1f} ms    {ar_naive_tps:<8.0f}")
print(f"  {'AR 64 (HF generate, KV cache)':<35} {ar_kv*1000:<8.1f} ms    {ar_kv_tps:<8.0f}")
print(f"  {'DifDecLM 64 (K='+str(K)+')':<35} {dif_gen*1000:<8.1f} ms    {dif_tps:<8.0f}")

speedup_naive = ar_naive / dif_gen
speedup_kv    = ar_kv / dif_gen
print(f"\n  Speedup (AR naive / DifDecLM):  {speedup_naive:.1f}x")
print(f"  Speedup (AR kv-cache / DifDecLM): {speedup_kv:.1f}x")

print(f"\n  AR output (KV cache): {ar_text_kv[:120]}")
print(f"  DifDec output:         {dif_text[:120]}")
print("\n  Note: DifDecLM head is randomly initialized (untrained).")
print("  Output quality will not resemble AR until trained.")
print("  Latency comparison IS valid - same backbone, same CPU/GPU.")
