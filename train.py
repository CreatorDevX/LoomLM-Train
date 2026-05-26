"""
train.py - DifDecLM training script (Colab-friendly).

All arguments exposed via CLI. Phase 1 (stabilize) by default.
"""

import os, sys, time, math, json, warnings
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from difdecLM import DifDecConfig
from difdecLM.model.difdec_lm import DifDecLM
from difdecLM.training import DiffusionProcess, DiffusionLoss
from difdecLM.data import collate_blocks

# ── CLI ──────────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser(description="DifDecLM Training")
# Mode
parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
parser.add_argument("--seed", type=int, default=42)

# Model architecture
parser.add_argument("--backbone", type=str, default="HuggingFaceTB/SmolLM2-135M")
parser.add_argument("--d-decoder", type=int, default=384)
parser.add_argument("--decoder-layers", type=int, default=6)
parser.add_argument("--decoder-heads", type=int, default=6)
parser.add_argument("--d-ff", type=int, default=1536)
parser.add_argument("--embedding-mode", type=str, default="shared", choices=["shared", "separate"])
parser.add_argument("--block-size", type=int, default=64)
parser.add_argument("--max-blocks", type=int, default=16)
parser.add_argument("--d-time-embed", type=int, default=384)

# Diffusion
parser.add_argument("--diffusion-steps", type=int, default=1000)
parser.add_argument("--sampling-steps", type=int, default=8)
parser.add_argument("--noise-schedule", type=str, default="cosine")
parser.add_argument("--prediction-type", type=str, default="epsilon")

# Training
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--grad-accum", type=int, default=2)
parser.add_argument("--max-steps", type=int, default=2000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr-min", type=float, default=1e-6)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--max-grad-norm", type=float, default=1.0)

# Loss weights
parser.add_argument("--diff-weight", type=float, default=1.0)
parser.add_argument("--clm-weight", type=float, default=0.0)
parser.add_argument("--eos-weight", type=float, default=0.01)
parser.add_argument("--clm-ramp-steps", type=int, default=0)

# Dataset
parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
parser.add_argument("--dataset-config", type=str, default="default")
parser.add_argument("--max-samples", type=int, default=50000)
parser.add_argument("--streaming", action="store_true", default=True)
parser.add_argument("--no-streaming", dest="streaming", action="store_false")
parser.add_argument("--num-workers", type=int, default=0)

# Logging & saving
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--eval-interval", type=int, default=200)
parser.add_argument("--save-interval", type=int, default=500)
parser.add_argument("--output-dir", type=str, default="checkpoints")
parser.add_argument("--no-wandb", action="store_true")
parser.add_argument("--wandb-project", type=str, default="difdeclm")

# Performance
parser.add_argument("--no-compile", action="store_true")
parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"])
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
parser.add_argument("--no-amp", action="store_true", help="disable autocast")

# Unfreeze
parser.add_argument("--unfreeze-layers", type=int, default=0)

args = parser.parse_args()

# ── Build Config ─────────────────────────────────────────────────────────────
config = DifDecConfig()
config.seed = args.seed
c = config

c.backbone.model_name = args.backbone
c.backbone.freeze = args.phase < 3
c.backbone.unfreeze_last_n_layers = args.unfreeze_layers if args.phase == 3 else 0

c.decoder.d_decoder = args.d_decoder
c.decoder.n_layers = args.decoder_layers
c.decoder.n_heads = args.decoder_heads
c.decoder.d_ff = args.d_ff
c.decoder.embedding_mode = args.embedding_mode

c.block.block_size = args.block_size
c.block.max_blocks = args.max_blocks
c.block.max_seq_len = args.block_size * args.max_blocks

c.diffusion.timesteps = args.diffusion_steps
c.diffusion.sampling_steps = args.sampling_steps
c.diffusion.noise_schedule = args.noise_schedule
c.diffusion.prediction_type = args.prediction_type
c.diffusion.d_time_embed = args.d_time_embed

c.training.batch_size = args.batch_size
c.training.gradient_accumulation_steps = args.grad_accum
c.training.max_steps = args.max_steps
c.training.warmup_steps = args.warmup_steps
c.training.lr = args.lr
c.training.lr_min = args.lr_min
c.training.weight_decay = args.weight_decay
c.training.max_grad_norm = args.max_grad_norm
c.training.diffusion_loss_weight = args.diff_weight
c.training.clm_loss_weight = args.clm_weight
c.training.eos_loss_weight = args.eos_weight
c.training.clm_loss_ramp_steps = args.clm_ramp_steps
c.training.log_interval = args.log_interval
c.training.eval_interval = args.eval_interval
c.training.save_interval = args.save_interval
c.training.output_dir = args.output_dir
c.training.phase = args.phase

c.data.dataset_name = args.dataset
c.data.dataset_config = args.dataset_config
c.data.max_samples = args.max_samples
c.data.streaming = args.streaming
c.data.num_workers = args.num_workers

c.device = "cuda"
c.dtype = args.dtype
c.compile = not args.no_compile

# Phase overrides
if args.phase == 1:
    c.backbone.freeze = True
    c.training.diffusion_loss_weight = 1.0
    c.training.clm_loss_weight = 0.0
    c.training.eos_loss_weight = 0.01
    c.training.lr = args.lr
elif args.phase == 2:
    c.backbone.freeze = True
    c.training.diffusion_loss_weight = 1.0
    c.training.clm_loss_weight = 0.3
    c.training.eos_loss_weight = 0.05
    c.training.clm_loss_ramp_steps = args.clm_ramp_steps or 3000
    c.training.lr = args.lr or 5e-5
elif args.phase == 3:
    c.backbone.freeze = False
    c.backbone.unfreeze_last_n_layers = args.unfreeze_layers or 4
    c.training.diffusion_loss_weight = 0.5
    c.training.clm_loss_weight = 0.5
    c.training.eos_loss_weight = 0.02
    c.training.clm_loss_ramp_steps = args.clm_ramp_steps or 2000
    c.training.lr = args.lr or 2e-5

SEQ_LEN = c.block.max_seq_len
N_BLOCKS = c.block.max_blocks
BLOCK_SIZE = c.block.block_size

# ── Determinism ──────────────────────────────────────────────────────────────
torch.manual_seed(c.seed)
np.random.seed(c.seed)

# ── Device / dtype ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
use_amp = not args.no_amp and device.type == "cuda" and amp_dtype != torch.float32
print(f"  Device: {device}  AMP: {amp_dtype if use_amp else 'off'}")

# ── Model ────────────────────────────────────────────────────────────────────
print("Building model...")
model = DifDecLM(c)
model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

if c.backbone.freeze:
    for p in model.backbone.parameters():
        p.requires_grad_(False)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,}  Total: {total:,}")

# ── Compile ──────────────────────────────────────────────────────────────────
if c.compile and hasattr(torch, "compile") and device.type == "cuda":
    print(f"  Compiling decoder head ({args.compile_mode})...")
    model.decoder = torch.compile(model.decoder, mode=args.compile_mode)
    model.time_embedding = torch.compile(model.time_embedding, mode=args.compile_mode)
    model.projection_head = torch.compile(model.projection_head, mode=args.compile_mode)
    if hasattr(model, "embed_proj"):
        model.embed_proj = torch.compile(model.embed_proj, mode=args.compile_mode)

# ── Diffusion components ─────────────────────────────────────────────────────
diff_process = DiffusionProcess(c).to(device)
loss_fn = DiffusionLoss(c)

# ── Optimizer ────────────────────────────────────────────────────────────────
trainable_params = model.get_trainable_params()
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=c.training.lr,
    betas=(c.training.adam_beta1, c.training.adam_beta2),
    eps=c.training.adam_eps,
    weight_decay=c.training.weight_decay,
)


def _cosine_lr(step, warmup, max_steps, lr, lr_min):
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return max(0.5 * (1.0 + math.cos(math.pi * progress)), lr_min / lr)


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda s: _cosine_lr(s, c.training.warmup_steps, c.training.max_steps, c.training.lr, c.training.lr_min),
)

# ── Dataset ──────────────────────────────────────────────────────────────────
from difdecLM.data import BlockDiffusionDataset
from torch.utils.data import DataLoader, IterableDataset


class FixedLengthDataset(IterableDataset):
    """Enforces every sample to exactly SEQ_LEN tokens (for fixed-shape compile)."""
    def __init__(self, config):
        self.base = BlockDiffusionDataset(config)
        self.seq_len = config.block.max_seq_len
        self.block_size = config.block.block_size
        self.pad_id = config.training.pad_token_id

    def __iter__(self):
        for item in self.base:
            seq_len = item["input_ids"].shape[0]
            n_blocks = item["block_tokens"].shape[0]
            target_blocks = self.seq_len // self.block_size
            if n_blocks < target_blocks:
                pad_len = self.seq_len - seq_len
                item["input_ids"] = torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_id, dtype=torch.long)])
                item["attention_mask"] = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                pad_blocks = target_blocks - n_blocks
                item["block_tokens"] = torch.cat([item["block_tokens"], torch.full((pad_blocks, self.block_size), self.pad_id, dtype=torch.long)])
            elif n_blocks > target_blocks:
                item["input_ids"] = item["input_ids"][:self.seq_len]
                item["attention_mask"] = item["attention_mask"][:self.seq_len]
                item["block_tokens"] = item["block_tokens"][:target_blocks]
            yield item


print("Creating dataset...")
train_dataset = FixedLengthDataset(c)
train_loader = DataLoader(
    train_dataset,
    batch_size=c.training.batch_size,
    shuffle=False,
    collate_fn=collate_blocks,
    num_workers=0,
    pin_memory=True,
)

# ── Validation set ───────────────────────────────────────────────────────────
EVAL_TEXTS = [
    "The transformer architecture uses self-attention to process sequences in parallel, enabling efficient training on large text corpora.",
    "Diffusion models gradually add noise to data and learn to reverse this process, generating high-quality samples from pure noise.",
    "Language models predict the next token given a sequence of previous tokens, forming the foundation of modern natural language processing.",
    "Gradient descent optimizes neural network parameters by computing gradients of the loss function with respect to each weight.",
    "Transfer learning pretrains a model on a large dataset before fine-tuning on a specific downstream task, improving sample efficiency.",
]


def build_eval_batch(texts, config, tokenizer):
    items = []
    for text in texts:
        tokens = tokenizer.encode(text, truncation=True, max_length=SEQ_LEN)
        total_len = ((len(tokens) - 1) // BLOCK_SIZE) * BLOCK_SIZE
        if total_len < BLOCK_SIZE:
            total_len = BLOCK_SIZE
        tokens = tokens[:total_len + 1]
        input_ids = tokens[:-1]
        block_tokens = tokens[1:]
        n_blocks = len(block_tokens) // BLOCK_SIZE
        input_ids = torch.tensor(input_ids[:n_blocks * BLOCK_SIZE], dtype=torch.long)
        block_tokens = torch.tensor(block_tokens[:n_blocks * BLOCK_SIZE], dtype=torch.long).view(-1, BLOCK_SIZE)
        attention_mask = torch.ones_like(input_ids)
        # Pad to fixed length
        target_blocks = N_BLOCKS
        if n_blocks < target_blocks:
            pad_seq = (target_blocks - n_blocks) * BLOCK_SIZE
            input_ids = torch.cat([input_ids, torch.full((pad_seq,), config.training.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_seq, dtype=torch.long)])
            block_tokens = torch.cat([block_tokens, torch.full((target_blocks - n_blocks, BLOCK_SIZE), config.training.pad_token_id, dtype=torch.long)])
        items.append({"input_ids": input_ids, "attention_mask": attention_mask, "block_tokens": block_tokens})
    return collate_blocks(items)


eval_tokenizer = train_dataset.base.tokenizer
eval_batch = build_eval_batch(EVAL_TEXTS, c, eval_tokenizer)
eval_input_ids = eval_batch["input_ids"].to(device)
eval_attn_mask = eval_batch["attention_mask"].to(device)
eval_block_tokens = eval_batch["block_tokens"].to(device)
eval_block_mask = eval_batch["block_mask"].to(device)

# ── Wandb ────────────────────────────────────────────────────────────────────
if not args.no_wandb:
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                "phase": args.phase, "backbone": args.backbone,
                "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                "max_steps": args.max_steps, "lr": args.lr,
                "seq_len": SEQ_LEN, "blocks": N_BLOCKS, "block_size": BLOCK_SIZE,
                "decoder_layers": args.decoder_layers, "d_decoder": args.d_decoder,
                "trainable": trainable, "total": total,
                "compile": not args.no_compile, "dtype": args.dtype,
                "dataset": args.dataset, "max_samples": args.max_samples,
            },
        )
        WANDB = True
    except Exception as e:
        print(f"  wandb init failed: {e}")
        WANDB = False
else:
    WANDB = False


@torch.no_grad()
def evaluate(step):
    model.eval()
    B_eval = eval_input_ids.shape[0]
    n_blocks_eval = N_BLOCKS
    timesteps = diff_process.get_timesteps(B_eval, n_blocks_eval, device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        outputs = model(eval_input_ids, timesteps, attention_mask=eval_attn_mask)
        loss_fn.set_step(step)
        _, metrics = loss_fn(outputs, eval_block_tokens, block_mask=eval_block_mask)
    model.train()
    return metrics


# ── Training Loop ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Phase {args.phase}  |  Steps: {args.max_steps}")
print(f"  Batch: {args.batch_size}  Accum: {args.grad_accum}  Eff: {args.batch_size * args.grad_accum}")
print(f"  Seq: {SEQ_LEN}  Blocks: {N_BLOCKS}  Block: {BLOCK_SIZE}")
print(f"  LR: {args.lr}  Warmup: {args.warmup_steps}  Weight decay: {args.weight_decay}")
print(f"  Diff weight: {c.training.diffusion_loss_weight}  CLM weight: {c.training.clm_loss_weight}  EOS weight: {c.training.eos_loss_weight}")
print(f"  Compile: {c.compile}  AMP: {use_amp}  Wandb: {WANDB}")
print(f"{'='*60}\n")

model.train()
step = 0
best_loss = float("inf")
os.makedirs(c.training.output_dir, exist_ok=True)

# Resume
if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt["step"]
    best_loss = ckpt.get("best_loss", float("inf"))
    print(f"  Resumed from step {step}")

t0 = time.time()
tokens_accum = 0
data_iter = iter(train_loader)

while step < c.training.max_steps:
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_metrics = {}
    t_step_start = time.time()

    for micro in range(c.training.gradient_accumulation_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        block_tokens = batch["block_tokens"].to(device, non_blocking=True)
        block_mask = batch.get("block_mask")
        if block_mask is not None:
            block_mask = block_mask.to(device, non_blocking=True)

        B, n_blocks, block_size = block_tokens.shape
        timesteps = diff_process.get_timesteps(B, n_blocks, device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(input_ids, timesteps, attention_mask=attention_mask)
            loss_fn.set_step(step)
            loss, metrics = loss_fn(outputs, block_tokens, block_mask=block_mask)
            loss = loss / c.training.gradient_accumulation_steps

        loss.backward()

        accum_loss += metrics["total_loss"]
        for k, v in metrics.items():
            if k not in accum_metrics:
                accum_metrics[k] = 0.0
            accum_metrics[k] += v

    # Gradient clipping & step
    torch.nn.utils.clip_grad_norm_(trainable_params, c.training.max_grad_norm)
    optimizer.step()
    scheduler.step()
    step += 1

    tokens_this_step = c.training.batch_size * N_BLOCKS * BLOCK_SIZE * c.training.gradient_accumulation_steps
    tokens_accum += tokens_this_step
    elapsed = time.time() - t_step_start
    tok_sec = tokens_this_step / elapsed if elapsed > 0 else 0

    for k in accum_metrics:
        accum_metrics[k] /= c.training.gradient_accumulation_steps

    lr_val = scheduler.get_last_lr()[0]

    # ── Log ──
    if step % c.training.log_interval == 0:
        print(
            f"  Step {step:4d}/{c.training.max_steps} | "
            f"Loss: {accum_metrics['total_loss']:.4f} | "
            f"Diff: {accum_metrics['diff_loss']:.4f} | "
            f"Token: {accum_metrics['token_loss']:.4f} | "
            f"EOS: {accum_metrics['eos_loss']:.4f} | "
            f"tok/s: {tok_sec:.0f} | "
            f"LR: {lr_val:.2e}"
        )

        if WANDB:
            wandb.log({
                "train/loss": accum_metrics["total_loss"],
                "train/diff_loss": accum_metrics["diff_loss"],
                "train/token_loss": accum_metrics["token_loss"],
                "train/eos_loss": accum_metrics["eos_loss"],
                "train/lr": lr_val,
                "train/tokens_per_sec": tok_sec,
                "train/tokens_total": tokens_accum,
            }, step=step)

    # ── Eval ──
    if step % c.training.eval_interval == 0:
        eval_metrics = evaluate(step)
        print(
            f"  ── Eval [{step}] ── "
            f"Loss: {eval_metrics['total_loss']:.4f} | "
            f"Diff: {eval_metrics['diff_loss']:.4f} | "
            f"Token: {eval_metrics['token_loss']:.4f} | "
            f"EOS: {eval_metrics['eos_loss']:.4f}"
        )
        if eval_metrics["total_loss"] < best_loss:
            best_loss = eval_metrics["total_loss"]
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss,
                        "config": c.to_dict()},
                       os.path.join(c.training.output_dir, "best.pt"))
            print(f"  ✓ New best model (loss={best_loss:.4f})")
        if WANDB:
            wandb.log({
                "eval/loss": eval_metrics["total_loss"],
                "eval/diff_loss": eval_metrics["diff_loss"],
                "eval/token_loss": eval_metrics["token_loss"],
                "eval/eos_loss": eval_metrics["eos_loss"],
                "eval/best_loss": best_loss,
            }, step=step)

    # ── Save ──
    if step % c.training.save_interval == 0:
        path = os.path.join(c.training.output_dir, f"checkpoint_step_{step}.pt")
        torch.save({
            "step": step, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss, "config": c.to_dict(),
        }, path)
        print(f"  ✓ Saved {path}")

# ── Final ──
elapsed_total = time.time() - t0
avg_tok_sec = tokens_accum / elapsed_total if elapsed_total > 0 else 0
print(f"\n{'='*60}")
print(f"  Training complete. Step: {step}  Time: {elapsed_total:.0f}s")
print(f"  Total tokens processed: {tokens_accum:,}  Avg tok/s: {avg_tok_sec:.0f}")
print(f"{'='*60}")

final_path = os.path.join(c.training.output_dir, "final.pt")
torch.save({
    "step": step, "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_loss": best_loss, "config": c.to_dict(),
}, final_path)
print(f"  Final model saved to {final_path}")

if WANDB:
    wandb.log({"train/avg_tokens_per_sec": avg_tok_sec, "train/total_tokens": tokens_accum}, step=step)
    wandb.finish()
