"""
Measure AR KV-cache scaling: latency, compute, and memory per token
as context grows from 64 to 2048 tokens.
Compare against DifDecLM diffusion (fixed cost).
"""
import sys; sys.path.insert(0, "difdecLM")
import os, time, torch, numpy as np

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"  Device: {device} ({gpu_name})  |  Torch {torch.__version__}")

def bytes_str(b):
    if b < 1024: return f"{b:.0f}B"
    if b < 1024**2: return f"{b/1024:.1f}KB"
    if b < 1024**3: return f"{b/1024**2:.1f}MB"
    return f"{b/1024**3:.2f}GB"

# ---- Load model ----
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
).to(device)
model.eval()
N_LAYERS = 30; N_HEADS = 9; HEAD_DIM = 64; D_MODEL = 576; DTYPE = 4
print(f"  {N_LAYERS} layers, {N_HEADS} heads, {HEAD_DIM} head_dim, {D_MODEL} d_model")

# Long sequence
prompt_text = "The future of artificial intelligence will transform "
tokens = tokenizer.encode(prompt_text)
while len(tokens) < 2048 + 64:
    tokens.extend(tokens[:min(128, 2048+64-len(tokens))])
tokens = tokens[:2048+64]
full_ids = torch.tensor([tokens], device=device)

CONTEXTS = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]

# ================================================================
#  PART 1: Per-token latency at each context length (KV cache)
# ================================================================
print("\n" + "="*70)
print("  PART 1: AR KV-CACHE PER-TOKEN LATENCY")
print("="*70)
print(f"  {'Context':>7} | {'1 token (ms)':>14} | {'64 extrap':>10} | {'KV cache':>10} | {'Attn %':>8}")
print(f"  {'-'*7} | {'-'*14} | {'-'*10} | {'-'*10} | {'-'*8}")

per_token_data = {}
for ctx in CONTEXTS:
    ids = full_ids[:, :ctx]
    next_id = full_ids[:, ctx:ctx+1]
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    out = model(input_ids=next_id, use_cache=True, past_key_values=past)
    past = out.past_key_values

    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        out = model(input_ids=next_id, use_cache=True, past_key_values=past)
        past = out.past_key_values
        _ = out.logits[:, -1:, :]
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    ms = float(np.mean(times)) * 1000
    kv = sum(t.numel()*t.element_size() for l in past for t in l)
    mlp = 2*D_MODEL*4*D_MODEL*N_LAYERS
    attn = 4*N_LAYERS*N_HEADS*HEAD_DIM*ctx
    pct = attn/(mlp+attn)*100
    per_token_data[ctx] = {"ms": ms, "kv": kv}
    print(f"  {ctx:>7} | {ms:>8.2f}  ms    | {ms*64/1000:>5.2f}s    | {bytes_str(kv):>8} | {pct:>5.1f}%")

# ================================================================
#  PART 2: AR: generate 64 tokens from key context lengths
# ================================================================
print("\n" + "="*70)
print("  PART 2: AR GENERATE 64 TOKENS (real end-to-end)")
print("="*70)
print(f"  {'Start ctx':>10} | {'Total':>10} | {'Per tok':>10} | {'Final KV':>12}")
print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")

ar_64_data = {}
for ctx in [64, 256, 512, 1024, 2048]:
    def gen(ct=ctx):
        with torch.no_grad():
            ids = full_ids[:, :ct].clone(); past = None
            for _ in range(64):
                if past is None:
                    out = model(input_ids=ids, use_cache=True)
                else:
                    out = model(input_ids=ids[:,-1:], use_cache=True, past_key_values=past)
                past = out.past_key_values
                ids = torch.cat([ids, out.logits[:,-1:,:].argmax(dim=-1)], dim=1)
            return ids, past
    gen(ctx)  # warmup
    t0 = time.perf_counter()
    ids_out, past_final = gen(ctx)
    if device.type == "cuda": torch.cuda.synchronize()
    t = time.perf_counter() - t0
    kv_final = sum(t.numel()*t.element_size() for l in past_final for t in l)
    ar_64_data[ctx] = (t, kv_final)
    print(f"  {ctx:>10} | {t*1000:>8.1f}ms  | {t/64*1000:>6.2f}ms   | {bytes_str(kv_final):>10}")

# ================================================================
#  PART 3: DifDecLM diffusion (fixed cost)
# ================================================================
print("\n" + "="*70)
print("  PART 3: DIFDECLM DIFFUSION (fixed cost)")
print("="*70)

from difdecLM.config import DifDecConfig
from difdecLM.model.time_embedding import TimeEmbedding
from difdecLM.model.diffusion_decoder import DiffusionDecoderStack

config = DifDecConfig()
d_back, d_dec, B, BLK = config.backbone.d_backbone, config.decoder.d_decoder, 1, config.block.block_size
embed_proj = torch.nn.Linear(d_back, d_dec).to(device)
time_emb = TimeEmbedding(config.diffusion.d_time_embed, use_mlp=True).to(device)
decoder = DiffusionDecoderStack(config).to(device)
emb_weight = model.get_input_embeddings().weight

T = config.diffusion.timesteps
s = config.diffusion.cosine_s
t_lin = torch.arange(T+1, device=device).float() / T
ab = torch.cos((t_lin + s)/(1.0+s)*(torch.pi/2.0)).clamp(min=0.0, max=1.0)

def difdec_gen(K):
    with torch.no_grad():
        h = model.model(full_ids[:, :64]).last_hidden_state
        ctx = h[:, -1, :]
        x = torch.randn(B, BLK, d_dec, device=device)
        steps = torch.linspace(T-1, 0, K, device=device).long()
        for i in range(K):
            t_val = int(steps[i].item())
            te = time_emb(torch.full((B,), t_val, device=device, dtype=torch.float))
            n = decoder(x, ctx, te)
            sa, s1a = ab[t_val].sqrt(), (1.0-ab[t_val]).clamp(min=0.0).sqrt()
            x0 = (x - s1a * n) / sa.clamp(min=1e-8)
            an = ab[int(steps[i+1].item())].sqrt() if i < K-1 else torch.tensor(1.0, device=device)
            sn = (1.0-an*an).clamp(min=0.0).sqrt() if i < K-1 else torch.tensor(0.0, device=device)
            x = an * x0 + sn * n
        return (x @ embed_proj.weight @ emb_weight.T).argmax(dim=-1)

dif_data = {}
for K in [1, 2, 4, 6, 8]:
    difdec_gen(K); difdec_gen(K)
    if device.type == "cuda": torch.cuda.synchronize()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        difdec_gen(K)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    ms = float(np.mean(times)) * 1000
    dif_data[K] = ms
    print(f"  DifDec K={K}: {ms:>8.2f} ms (fixed, 64-token block)")

# ================================================================
#  FINAL TABLE
# ================================================================
print("\n" + "="*70)
print("  COMPARISON: GENERATE 64 TOKENS")
print("="*70)
print(f"  {'Method':<30} | {'Latency':>10} | {'Tok/s':>7} | {'Memory':>10} | {'Cost vs ctx':>15}")
print(f"  {'-'*30} | {'-'*10} | {'-'*7} | {'-'*10} | {'-'*15}")
for ctx in [64, 256, 512, 1024, 2048]:
    t, kv = ar_64_data[ctx]
    print(f"  {'AR (start='+str(ctx)+')':<30} | {t*1000:>7.1f}ms  | {64/t:>5.0f}  | {bytes_str(kv):>8} | {'O(ctx) grows':>15}")
for K in [1, 2, 4, 6, 8]:
    print(f"  {'DifDec K='+str(K):<30} | {dif_data[K]:>7.1f}ms  | {64/(dif_data[K]/1000):>5.0f}  | n/a       | {'O(1) flat':>15}")

# Big picture summary
print("\n" + "="*70)
print("  BIG PICTURE: SCALING TO 2048+ TOKENS")
print("="*70)

# Estimate time for AR to generate 2048 tokens from scratch
t_first_chunk = ar_64_data[64][0]  # first 64 tokens
# Estimate total: sum of 64-token chunks at increasing context lengths
# AR total = sum_{i=0}^{31} time_to_generate_64_at_context(64*i)
# Approximate using per_token_data
ar_total_2048 = 0
for i in range(32):  # 32 chunks of 64
    ctx = 64 + i * 64
    # find nearest measured context
    nearest = min(CONTEXTS, key=lambda c: abs(c-ctx))
    tok_ms = per_token_data[nearest]["ms"]
    ar_total_2048 += tok_ms * 64
print(f"  AR generate 2048 tokens (estimated): {ar_total_2048/1000:.1f}s")
print(f"  DifDec K=8 generate 2048 tokens (32 blocks): {dif_data[8]/1000*32:.1f}s")
print(f"  DifDec K=4 generate 2048 tokens (32 blocks): {dif_data[4]/1000*32:.1f}s")
print(f"  DifDec K=1 generate 2048 tokens (32 blocks): {dif_data[1]/1000*32:.1f}s")

# KV cache size at 2048
kv_2048 = sum(t.numel()*t.element_size() for l in past_final for t in l)
print(f"\n  KV cache at 2048 context: {bytes_str(kv_2048)}")
print(f"  KV cache at 8192 context (extrap): {bytes_str(kv_2048 * 4)}")
print(f"  AR per-token compute at 2048: {per_token_data[2048]['ms']*1000:.0f}us")
