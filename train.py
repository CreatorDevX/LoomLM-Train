"""
train.py - LoomLM (DifDecLM) training script. All CLI args exposed.
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

import argparse
parser = argparse.ArgumentParser(description="LoomLM Training")
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
parser.add_argument("--sampling-steps", type=int, default=4)
parser.add_argument("--noise-schedule", type=str, default="cosine")
parser.add_argument("--prediction-type", type=str, default="x0")

# Training
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--grad-accum", type=int, default=2)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--lr-min", type=float, default=1e-6)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--max-grad-norm", type=float, default=1.0)

# Loss weights
parser.add_argument("--diff-weight", type=float, default=None)
parser.add_argument("--clm-weight", type=float, default=None)
parser.add_argument("--consistency-weight", type=float, default=None)
parser.add_argument("--clm-ramp-steps", type=int, default=0)

# Finetuning mode
parser.add_argument("--full-ft", action="store_true", help="Full finetuning (unfreeze backbone, no LoRA)")
parser.add_argument("--use-lora", action="store_true", default=False, help="Apply LoRA to backbone")
parser.add_argument("--lora-r", type=int, default=32)
parser.add_argument("--lora-alpha", type=int, default=32)


# Dataset
parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
parser.add_argument("--dataset-config", type=str, default="default")
parser.add_argument("--max-samples", type=int, default=100000)
parser.add_argument("--streaming", action="store_true", default=True)
parser.add_argument("--no-streaming", dest="streaming", action="store_false")
parser.add_argument("--num-workers", type=int, default=0)

# Logging & saving
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--inference-interval", type=int, default=125)
parser.add_argument("--eval-interval", type=int, default=200)
parser.add_argument("--save-interval", type=int, default=500)
parser.add_argument("--output-dir", type=str, default="checkpoints")
parser.add_argument("--no-wandb", action="store_true")
parser.add_argument("--wandb-project", type=str, default="loomlm")

# Performance
parser.add_argument("--no-compile", action="store_true")
parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"])
parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"])
parser.add_argument("--no-amp", action="store_true")

# Multi-GPU
parser.add_argument("--num-processes", type=int, default=1, help="Number of processes for notebook_launcher (DDP)")


def main(args):
    # ── DDP ──────────────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1 and torch.cuda.is_available()
    if is_ddp:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = (not is_ddp) or (local_rank == 0)

    # ── Build Config ─────────────────────────────────────────────────────────────
    config = DifDecConfig()
    config.seed = args.seed
    c = config

    c.backbone.model_name = args.backbone
    c.backbone.freeze = True
    c.backbone.unfreeze_last_n_layers = 0

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
    c.training.warmup_steps = args.warmup_steps
    c.training.lr_min = args.lr_min
    c.training.weight_decay = args.weight_decay
    c.training.max_grad_norm = args.max_grad_norm
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

    # Phase presets (loss schedule only — backbone mode controlled separately)
    if args.phase == 1:
        c.training.max_steps = args.max_steps or 2000
        c.training.lr = args.lr or 1e-4
        c.training.diffusion_loss_weight = args.diff_weight or 1.0
        clm_w = args.clm_weight or 0.3
        c.training.clm_loss_weight = clm_w
        c.training.clm_loss_max_weight = clm_w
        c.training.consistency_loss_weight = args.consistency_weight or 0.1
        c.training.clm_loss_ramp_steps = args.clm_ramp_steps or 500
    elif args.phase == 3:
        c.training.max_steps = args.max_steps or 3000
        c.training.lr = args.lr or 2e-5
        c.training.diffusion_loss_weight = args.diff_weight or 0.5
        c.training.clm_loss_weight = args.clm_weight or 0.5
        c.training.consistency_loss_weight = args.consistency_weight or 0.0
        c.training.clm_loss_ramp_steps = args.clm_ramp_steps or 2000

    # Backbone training mode (independent of phase)
    if args.full_ft:
        c.backbone.freeze = False
        c.backbone.unfreeze_last_n_layers = 4
    else:
        c.backbone.freeze = True
        c.backbone.unfreeze_last_n_layers = 0
    DO_LORA = args.use_lora and not args.full_ft

    SEQ_LEN = c.block.max_seq_len
    N_BLOCKS = c.block.max_blocks
    BLOCK_SIZE = c.block.block_size

    # ── Determinism ──────────────────────────────────────────────────────────────
    seed = c.seed + (local_rank if is_ddp else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Device / dtype ───────────────────────────────────────────────────────────
    amp_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    use_amp = not args.no_amp and device.type == "cuda" and amp_dtype != torch.float32
    if is_main:
        print(f"  Device: {device}  AMP: {amp_dtype if use_amp else 'off'}  DDP: {is_ddp} (rank {local_rank}/{world_size})")

    # ── Model ────────────────────────────────────────────────────────────────────
    if is_main:
        print("Building model...")
    model = DifDecLM(c)
    model.to(device=device)
    if use_amp:
        model.to(dtype=amp_dtype)
        for m in model.decoder.modules():
            if isinstance(m, nn.LayerNorm):
                m.to(dtype=torch.float32)
        model.time_embedding.to(dtype=torch.float32)
    else:
        model.backbone.to(dtype=torch.float32)
        if c.decoder.embedding_mode == "shared" and hasattr(model.projection_head, 'embedding_weight'):
            model.projection_head.embedding_weight = model.backbone.get_embedding_matrix()

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    if c.backbone.freeze:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    if DO_LORA:
        if is_main:
            print(f"  Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}) to backbone + embeddings...")
        model.backbone.apply_lora(r=args.lora_r, alpha=args.lora_alpha)
        # Bridge LoRA embedding params to projection head for dense CLM gradient
        if hasattr(model.projection_head, 'attach_lora_embedding') and hasattr(model.backbone.embed_fn, 'lora_A'):
            model.projection_head.attach_lora_embedding(
                model.backbone.embed_fn.lora_A,
                model.backbone.embed_fn.lora_B,
                model.backbone.embed_fn.scaling,
            )

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,}  Total: {total:,}")
        # ── Compile ──
        if c.compile and hasattr(torch, "compile") and device.type == "cuda":
            print(f"  Compiling decoder...")
            model.decoder = torch.compile(model.decoder, options={"cudagraphs": False})

    # ── DDP wrap ─────────────────────────────────────────────────────────────────
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ── Diffusion components ─────────────────────────────────────────────────────
    diff_process = DiffusionProcess(c).to(device)
    loss_fn = DiffusionLoss(c)

    # ── Optimizer ────────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
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

    if is_main:
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

    # ── Validation set (from Fineweb-edu) ────────────────────────────────────────
    if is_main:
        print("Building validation batch from Fineweb-edu...")
    val_items = []
    val_iter = iter(train_dataset)
    for _ in range(c.training.batch_size):
        try:
            val_items.append(next(val_iter))
        except StopIteration:
            break
    val_batch = collate_blocks(val_items)
    eval_input_ids = val_batch["input_ids"].to(device)
    eval_attn_mask = val_batch["attention_mask"].to(device)
    eval_block_tokens = val_batch["block_tokens"].to(device)
    eval_block_mask = val_batch["block_mask"].to(device)

    # ── Inference generator ──────────────────────────────────────────────────────
    from difdecLM.inference import BlockGenerator

    INFERENCE_PROMPT = "The future of artificial intelligence"
    INFERENCE_MAX_BLOCKS = 4
    if is_main:
        inference_generator = BlockGenerator(model, c, device)

    # ── Wandb ────────────────────────────────────────────────────────────────────
    WANDB = False
    if is_main and not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config={
                    "phase": args.phase, "backbone": args.backbone,
                    "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                    "max_steps": c.training.max_steps, "lr": c.training.lr,
                    "seq_len": SEQ_LEN, "blocks": N_BLOCKS, "block_size": BLOCK_SIZE,
                    "decoder_layers": args.decoder_layers, "d_decoder": args.d_decoder,
                    "compile": not args.no_compile, "dtype": args.dtype,
                    "dataset": args.dataset, "max_samples": args.max_samples,
                    "sampling_steps": args.sampling_steps,
                    "use_lora": DO_LORA, "lora_r": args.lora_r,
                },
            )
            WANDB = True
        except Exception as e:
            print(f"  wandb init failed: {e}")

    # ── Evaluate ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(step):
        model.eval()
        timesteps = diff_process.get_timesteps(eval_input_ids.size(0), N_BLOCKS, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(eval_input_ids, timesteps, attention_mask=eval_attn_mask)
        loss_fn.set_step(step)
        _, metrics = loss_fn(outputs, eval_block_tokens, block_mask=eval_block_mask)
        model.train()
        return metrics

    # ── Inference example ────────────────────────────────────────────────────────
    @torch.no_grad()
    def run_inference_example(step):
        model.eval()
        try:
            text, ids = inference_generator.generate(
                INFERENCE_PROMPT, max_new_blocks=INFERENCE_MAX_BLOCKS,
                temperature=0.8, top_k=40, top_p=0.9,
            )
            print(f"\n  ── Gen [{step}] ── {text[:200]}\n")
            if WANDB:
                wandb.log({"inference/text": wandb.Html(f"<pre>{text}</pre>"), "inference/step": step}, step=step)
        except Exception as e:
            print(f"  Inference failed at step {step}: {e}")
        model.train()

    # ── Training Loop ────────────────────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*60}")
        print(f"  LoomLM  Phase {args.phase}  |  Steps: {c.training.max_steps}")
        print(f"  Batch: {args.batch_size}  Accum: {args.grad_accum}  Eff: {args.batch_size * args.grad_accum}")
        print(f"  Seq: {SEQ_LEN}  Blocks: {N_BLOCKS}  Block: {BLOCK_SIZE}")
        print(f"  LR: {c.training.lr}  Warmup: {args.warmup_steps}  Weight decay: {args.weight_decay}")
        print(f"  Diff: {c.training.diffusion_loss_weight}  CLM: {c.training.clm_loss_weight}  Cons: {c.training.consistency_loss_weight}")
        print(f"  Compile: {c.compile}  AMP: {use_amp}  Wandb: {WANDB}  LoRA: {DO_LORA}")
        print(f"{'='*60}\n")

    model.train()
    step = 0
    best_loss = float("inf")
    os.makedirs(c.training.output_dir, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if is_main:
            print(f"  Resumed from step {step}")

    t0 = time.time()
    tokens_accum = 0
    data_iter = iter(train_loader)

    while step < c.training.max_steps:
        optimizer.zero_grad()
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
            scaler.scale(loss).backward()

            for k, v in metrics.items():
                if k not in accum_metrics:
                    accum_metrics[k] = 0.0
                accum_metrics[k] += v

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, c.training.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
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
        if step % c.training.log_interval == 0 and is_main:
            pred_norm = accum_metrics.get("pred_norm", None)
            cons = accum_metrics.get("consistency_loss", 0.0)
            log_line = (
                f"  Step {step:4d}/{c.training.max_steps} | "
                f"Loss: {accum_metrics['total_loss']:.4f} | "
                f"Diff: {accum_metrics['diff_loss']:.4f} | "
                f"Token: {accum_metrics['token_loss']:.4f} | "
                f"Cons: {cons:.4f} | "
                f"tok/s: {tok_sec:.0f} | "
                f"LR: {lr_val:.2e}"
            )
            if pred_norm is not None:
                log_line += f" | pred_norm: {pred_norm:.3f}"
            print(log_line)
            if WANDB:
                wandb_log = {
                    "train/loss": accum_metrics["total_loss"],
                    "train/diff_loss": accum_metrics["diff_loss"],
                    "train/token_loss": accum_metrics["token_loss"],
                    "train/lr": lr_val,
                    "train/tokens_per_sec": tok_sec,
                    "train/tokens_total": tokens_accum,
                    "train/clm_weight": loss_fn.clm_weight,
                }
                cons = accum_metrics.get("consistency_loss", 0)
                if cons != 0:
                    wandb_log["train/consistency_loss"] = cons
                if pred_norm is not None:
                    target_norm = accum_metrics.get("target_norm", 0)
                    wandb_log["train/pred_norm"] = pred_norm
                    wandb_log["train/target_norm"] = target_norm
                wandb.log(wandb_log, step=step)

        # ── Eval ──
        if step % c.training.eval_interval == 0:
            eval_metrics = evaluate(step)
            if is_main:
                print(
                    f"  ── Eval [{step}] ── "
                    f"Loss: {eval_metrics['total_loss']:.4f} | "
                    f"Diff: {eval_metrics['diff_loss']:.4f} | "
                    f"Token: {eval_metrics['token_loss']:.4f}"
                )
                if eval_metrics["total_loss"] < best_loss:
                    best_loss = eval_metrics["total_loss"]
                    torch.save({
                        "step": step, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_loss,
                        "config": c.to_dict(),
                    }, os.path.join(c.training.output_dir, "best.pt"))
                    print(f"  ✓ New best (loss={best_loss:.4f})")
                if WANDB:
                    wandb.log({
                        "eval/loss": eval_metrics["total_loss"],
                        "eval/diff_loss": eval_metrics["diff_loss"],
                        "eval/token_loss": eval_metrics["token_loss"],
                        "eval/best_loss": best_loss,
                    }, step=step)

        # ── Diagnostic: norm check at step 100 ──
        if step == 100 and accum_metrics.get("pred_norm") is not None and is_main:
            pn = accum_metrics["pred_norm"]
            tn = accum_metrics["target_norm"]
            ratio = pn / max(tn, 1e-8)
            if ratio < 0.3:
                print(f"  ⚠ pred_norm={pn:.3f} vs target_norm={tn:.3f} (ratio={ratio:.2f}) — "
                      f"head may be predicting near-zero. Consider --no-amp or --prediction-type epsilon.")
            else:
                print(f"  ✓ Norm check: pred={pn:.3f}, target={tn:.3f}, ratio={ratio:.2f}")

        # ── Inference example ──
        if step % args.inference_interval == 0 and is_main:
            run_inference_example(step)

        # ── Save ──
        if step % c.training.save_interval == 0 and is_main:
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
    if is_main:
        print(f"\n{'='*60}")
        print(f"  LoomLM Phase {args.phase} complete. Step: {step}  Time: {elapsed_total:.0f}s")
        print(f"  Total tokens: {tokens_accum:,}  Avg tok/s: {avg_tok_sec:.0f}")
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


if __name__ == "__main__":
    raw_args = parser.parse_args()
    if raw_args.num_processes > 1:
        from accelerate import notebook_launcher
        notebook_launcher(main, (raw_args,), num_processes=raw_args.num_processes)
    else:
        main(raw_args)
