"""
Sweep K = 1,2,4,6,8 for DifDecLM diffusion vs AR baselines (KV cache + naive).
Compares latency and throughput for generating 64 tokens with each setting.
"""
import sys; sys.path.insert(0, "difdecLM")
import os, time, torch, torch.nn as nn, numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"  Device: {device} ({gpu_name})  |  Torch {torch.__version__}")

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
    mean_s = float(np.mean(times)); std_s = float(np.std(times))
    print(f"  {desc:<45} {mean_s*1000:8.1f} +/- {std_s*1000:.1f} ms")
    return mean_s, std_s

# ---- Load backbone ----
print("\n" + "="*60)
print("  Loading SmolLM2-135M...")
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
base_model = backbone_lm.model
emb_weight = backbone_lm.get_input_embeddings().weight
print(f"  Backbone params: {backbone_params:,}")

# ---- Build decoder head ----
from difdecLM.config import DifDecConfig
from difdecLM.model.time_embedding import TimeEmbedding
from difdecLM.model.diffusion_decoder import DiffusionDecoderStack

config = DifDecConfig()
d_back = config.backbone.d_backbone
d_dec  = config.decoder.d_decoder
V      = config.vocab_size
B      = 1
BLOCK  = config.block.block_size

embed_proj = nn.Linear(d_back, d_dec).to(device)
time_emb   = TimeEmbedding(config.diffusion.d_time_embed, use_mlp=True).to(device)
decoder    = DiffusionDecoderStack(config).to(device)

# ---- Precompute schedule once ----
T = config.diffusion.timesteps
s = config.diffusion.cosine_s
t_lin = torch.arange(T + 1, device=device).float() / T
f_cos = torch.cos((t_lin + s) / (1.0 + s) * (torch.pi / 2.0))
alpha_bar = f_cos.clamp(min=0.0, max=1.0)

def get_noise_pred(noise_pred_raw, E_t, sqrt_ab, sqrt_1ab):
    ptype = config.diffusion.prediction_type
    if ptype == "epsilon":
        return noise_pred_raw
    elif ptype == "x0":
        return (E_t - sqrt_ab * noise_pred_raw) / sqrt_1ab.clamp(min=1e-8)
    elif ptype == "v":
        return sqrt_ab * noise_pred_raw + sqrt_1ab * E_t
    return noise_pred_raw

def difdec_generate(K):
    """Run diffusion generation with exactly K DDIM steps."""
    with torch.no_grad():
        hidden = base_model(prompt_ids).last_hidden_state
        ctx = hidden[:, -1:, :].squeeze(1)
        x_t = torch.randn(B, BLOCK, d_dec, device=device)
        steps = torch.linspace(T - 1, 0, K, device=device).long()
        for i in range(K):
            t_val = int(steps[i].item())
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.float)
            t_emb_val = time_emb(t_batch)
            noise_raw = decoder(x_t, ctx, t_emb_val)
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
        h_up = x_t @ embed_proj.weight
        logits = h_up @ emb_weight.T
        tokens = logits.argmax(dim=-1)
    return tokens

# ---- Prompt ----
prompt_text = "The future of artificial intelligence will transform"
prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
prompt_len = prompt_ids.shape[1]

# ---- AR baselines ----
print("\n" + "-"*60)
print("  AR BASELINES")
print("-"*60)

def ar_naive():
    with torch.no_grad():
        ids = prompt_ids.clone()
        for _ in range(64):
            logits = backbone_lm(ids).logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    return ids

def ar_kvcache():
    with torch.no_grad():
        out = backbone_lm.generate(
            prompt_ids, max_new_tokens=64, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.pad_token_id,
        )
    return out

ar_n, ar_n_std = timed(ar_naive,    n_warmup=1, n_trials=3, desc="AR 64 (naive, full fwd)")
ar_k, ar_k_std = timed(ar_kvcache,  n_warmup=1, n_trials=3, desc="AR 64 (HF generate, KV cache)")

ar_out = ar_kvcache()
ar_text = tokenizer.decode(ar_out[0, prompt_len:], skip_special_tokens=True)
print(f"  AR sample: {ar_text[:100]}...")

# ---- K sweep ----
print("\n" + "-"*60)
print("  DIFFUSION K-SWEEP")
print("-"*60)

results = {}
for K in [1, 2, 4, 6, 8]:
    fn = lambda K=K: difdec_generate(K)
    mean_s, std_s = timed(fn, n_warmup=1, n_trials=5, desc=f"DifDec K={K}  64 tokens")

    out = difdec_generate(K)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    results[K] = {"mean_s": mean_s, "std_s": std_s, "sample": text[:80]}
    print(f"  Sample K={K}: {text[:80]}...")

# ---- Summary ----
print("\n" + "="*60)
print("  FINAL COMPARISON")
print("="*60)
print(f"  {'Method':<30} {'Latency':<15} {'Tok/s':<10} {'vs AR-naive':<15} {'vs AR-kv':<15}")
print(f"  {'-'*30} {'-'*15} {'-'*10} {'-'*15} {'-'*15}")
print(f"  {'AR naive':<30} {ar_n*1000:<8.1f} ms    {64/ar_n:<8.0f} {'1.0x':<15} {'-':<15}")
print(f"  {'AR KV cache':<30} {ar_k*1000:<8.1f} ms    {64/ar_k:<8.0f} {ar_n/ar_k:<5.1f}x{'':<10} {'1.0x':<15}")
for K in [1, 2, 4, 6, 8]:
    r = results[K]
    sp_n = ar_n / r["mean_s"]
    sp_k = ar_k / r["mean_s"]
    print(f"  {'DifDecLM K='+str(K):<30} {r['mean_s']*1000:<8.1f} ms    {64/r['mean_s']:<8.0f} {sp_n:<5.1f}x{'':<10} {sp_k:<5.1f}x{'':<10}")

print("\n  AR sample:    ", ar_text[:120])
for K in [1, 2, 4, 6, 8]:
    print(f"  DifDec K={K}: ", results[K]["sample"][:120])
print("\n  (DifDec head is randomly initialized - quality comparison not meaningful)")
