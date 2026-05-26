import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        tc = config.training
        self.diff_weight = tc.diffusion_loss_weight
        self.clm_weight = tc.clm_loss_weight
        self.consistency_weight = tc.consistency_loss_weight
        self.eos_weight = tc.eos_loss_weight
        self.clm_ramp_steps = tc.clm_loss_ramp_steps
        self.clm_max_weight = tc.clm_loss_max_weight
        self.mask_pad = tc.mask_pad_positions
        self.pad_id = tc.pad_token_id

        self.prediction_type = config.diffusion.prediction_type
        self.timestep_weighting = config.diffusion.diffusion_timestep_weighting
        self.T = config.diffusion.timesteps

    def set_step(self, step):
        if self.clm_ramp_steps > 0:
            progress = min(step / self.clm_ramp_steps, 1.0)
            self.clm_weight = self.clm_max_weight * progress

    def _get_timestep_weights(self, timesteps):
        if self.timestep_weighting == "uniform" or timesteps is None:
            return 1.0
        elif self.timestep_weighting == "mid_weighted":
            t_norm = timesteps.float() / self.T
            weight = 1.0 - 2.0 * (t_norm - 0.5).abs()
            return weight.clamp(min=0.1)

    def forward(self, model_output, block_tokens, block_mask=None):
        B, n_blocks, block_size = block_tokens.shape

        # ── Diffusion loss ──────────────────────────────────────────────────
        if self.prediction_type == "x0":
            pred = model_output["pred_embeddings"].float()
            target = model_output["clean_embeddings"].float()
        else:
            pred = model_output["noise_pred"].float()
            target = model_output["noise_target"].float()

        diff_loss_per_block = F.mse_loss(pred, target, reduction="none").mean(dim=(-1, -2))
        tw = self._get_timestep_weights(model_output.get("timesteps"))

        if block_mask is not None:
            bm = block_mask.float()
            diff_loss = (diff_loss_per_block * bm * tw).sum() / bm.sum().clamp(min=1.0)
        else:
            diff_loss = diff_loss_per_block.mean() if isinstance(tw, float) else (diff_loss_per_block * tw).mean()

        # ── CLM loss ────────────────────────────────────────────────────────
        logits = model_output["logits"].float()
        V = logits.size(-1)
        token_loss = F.cross_entropy(
            logits.view(-1, V),
            block_tokens.view(-1),
            reduction="none",
        ).view(B, n_blocks, block_size)

        if self.mask_pad:
            pad_mask = (block_tokens != self.pad_id).float()
            if block_mask is not None:
                pad_mask = pad_mask * block_mask.unsqueeze(-1).float()
            token_loss = (token_loss * pad_mask).sum() / pad_mask.sum().clamp(min=1.0)
        else:
            token_loss = token_loss.mean()

        # ── EOS loss ─────────────────────────────────────────────────────────
        if self.eos_weight > 0:
            eos_logits = model_output["eos_logits"].float()
            eos_target = (block_tokens == self.pad_id).float()
            eos_loss_per_pos = F.binary_cross_entropy_with_logits(eos_logits, eos_target, reduction="none")
            if self.mask_pad:
                if block_mask is not None:
                    eos_loss_per_pos = eos_loss_per_pos * block_mask.unsqueeze(-1).float()
                eos_loss = eos_loss_per_pos.sum() / max(eos_loss_per_pos.numel(), 1)
            else:
                eos_loss = eos_loss_per_pos.mean()
        else:
            eos_loss = torch.tensor(0.0, device=block_tokens.device)

        # ── Consistency loss (embedding reconstruction) ────────────────────
        if self.consistency_weight > 0 and self.prediction_type == "epsilon":
            pred_emb = model_output["pred_embeddings"].float()
            clean_emb = model_output["clean_embeddings"].float()
            if block_mask is not None:
                bm = block_mask.unsqueeze(-1).unsqueeze(-1).float()
                consistency_loss = F.mse_loss(pred_emb * bm, clean_emb.detach() * bm, reduction="sum")
                consistency_loss = consistency_loss / bm.sum().clamp(min=1.0)
            else:
                consistency_loss = F.mse_loss(pred_emb, clean_emb.detach(), reduction="mean")
        else:
            consistency_loss = torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else block_tokens.device)

        # ── Total ───────────────────────────────────────────────────────────
        total_loss = (
            self.diff_weight * diff_loss
            + self.clm_weight * token_loss
            + self.consistency_weight * consistency_loss
            + self.eos_weight * eos_loss
        )

        metrics = {
            "diff_loss": diff_loss.detach().item(),
            "token_loss": token_loss.detach().item(),
            "consistency_loss": consistency_loss.detach().item(),
            "eos_loss": eos_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "clm_weight": self.clm_weight,
        }

        # Diagnostic norms (silent, stored for logging)
        if self.prediction_type == "x0":
            pred_norm = pred.detach().norm(dim=-1).mean().item()
            target_norm = target.detach().norm(dim=-1).mean().item()
            metrics["pred_norm"] = pred_norm
            metrics["target_norm"] = target_norm

        return total_loss, metrics
