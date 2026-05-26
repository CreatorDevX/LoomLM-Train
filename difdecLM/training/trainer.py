import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from .diffusion_process import DiffusionProcess
from .losses import DiffusionLoss


def _cosine_schedule(step, warmup_steps, max_steps, lr, lr_min):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
    return max(cosine_decay.item(), lr_min / lr)


class Trainer:
    def __init__(self, model, config, device="cuda"):
        self.model = model
        self.config = config
        self.device = device

        tc = config.training
        self.max_steps = tc.max_steps
        self.grad_accum = tc.gradient_accumulation_steps
        self.log_interval = tc.log_interval
        self.eval_interval = tc.eval_interval
        self.save_interval = tc.save_interval
        self.output_dir = tc.output_dir
        self.max_grad_norm = tc.max_grad_norm
        self.phase = tc.phase

        self.diff_process = DiffusionProcess(config).to(device)
        self.loss_fn = DiffusionLoss(config)

        trainable_params = model.get_trainable_params()
        self.optimizer = AdamW(
            trainable_params,
            lr=tc.lr,
            betas=(tc.adam_beta1, tc.adam_beta2),
            eps=tc.adam_eps,
            weight_decay=tc.weight_decay,
        )

        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _cosine_schedule(step, tc.warmup_steps, tc.max_steps, tc.lr, tc.lr_min),
        )

        self.step = 0
        self.best_loss = float("inf")

        os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.output_dir, f"checkpoint_step_{self.step}.pt")
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.to_dict(),
        }, path)
        print(f"  Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt["step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        print(f"  Loaded checkpoint from {path} (step {self.step})")

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        metrics_sum = {}
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if self.step >= self.max_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            block_tokens = batch["block_tokens"].to(self.device)
            block_mask = batch.get("block_mask", None)

            B, n_blocks, block_size = block_tokens.shape
            timesteps = self.diff_process.get_timesteps(B, n_blocks, self.device)

            model_output = self.model(input_ids, timesteps, attention_mask=attention_mask)

            self.loss_fn.set_step(self.step)
            loss, metrics = self.loss_fn(model_output, block_tokens, block_mask=block_mask)
            loss = loss / self.grad_accum

            loss.backward()

            total_loss += metrics["total_loss"]
            for k, v in metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += v
            n_batches += 1

            if (batch_idx + 1) % self.grad_accum == 0:
                clip_grad_norm_(self.model.get_trainable_params(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step += 1

                if self.step % self.log_interval == 0:
                    avg_loss = total_loss / n_batches
                    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"  Step {self.step}/{self.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Diff: {avg_metrics.get('diff_loss', 0):.4f} | "
                        f"Token: {avg_metrics.get('token_loss', 0):.4f} | "
                        f"EOS: {avg_metrics.get('eos_loss', 0):.4f} | "
                        f"LR: {lr:.2e}"
                    )
                    total_loss = 0.0
                    metrics_sum = {}
                    n_batches = 0

                if self.step % self.save_interval == 0:
                    self.save_checkpoint()

    def train(self, dataloader):
        print(f"\n{'='*60}")
        print(f"  Phase {self.phase} Training")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Trainable params: {self.model.get_num_trainable():,}")
        print(f"{'='*60}\n")

        while self.step < self.max_steps:
            self.train_epoch(dataloader)

        self.save_checkpoint(os.path.join(self.output_dir, "final.pt"))
        print(f"\n  Training complete. Final step: {self.step}")
