import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.training
        self.diff_weight = config.training.diffusion_loss_weight
        self.clm_weight = config.training.clm_loss_weight
        self.eos_weight = config.training.eos_loss_weight
        self.clm_ramp_steps = config.training.clm_loss_ramp_steps
        self.clm_max_weight = config.training.clm_loss_max_weight
        self.mask_pad = config.training.mask_pad_positions
        self.pad_id = config.training.pad_token_id
        self.block_size = config.block.block_size

    def set_step(self, step):
        if self.clm_ramp_steps > 0:
            progress = min(step / self.clm_ramp_steps, 1.0)
            self.clm_weight = self.clm_max_weight * progress

    def forward(self, model_output, block_tokens, block_mask=None):
        noise_pred = model_output["noise_pred"].float()
        noise_target = model_output["noise_target"].float()
        logits = model_output["logits"].float()
        eos_logits = model_output["eos_logits"].float()

        B, n_blocks, block_size, V = logits.shape

        if block_mask is not None:
            bm = block_mask.unsqueeze(-1).unsqueeze(-1).float()
            diff_loss = F.mse_loss(noise_pred * bm, noise_target * bm, reduction="sum")
            diff_loss = diff_loss / bm.sum().clamp(min=1.0)
        else:
            diff_loss = F.mse_loss(noise_pred, noise_target, reduction="mean")

        token_loss = F.cross_entropy(
            logits.view(-1, V),
            block_tokens.view(-1),
            reduction="none",
        ).view(B, n_blocks, block_size)

        if self.mask_pad:
            pad_mask = (block_tokens != self.pad_id).float()
            if block_mask is not None:
                pad_mask = pad_mask * block_mask.unsqueeze(-1).float()
            token_loss = token_loss * pad_mask
            token_loss = token_loss.sum() / pad_mask.sum().clamp(min=1.0)
        else:
            token_loss = token_loss.mean()

        eos_target = torch.zeros(B, n_blocks, block_size, device=eos_logits.device)
        eos_target[:, :, -1] = 1.0
        if block_mask is not None:
            eos_loss = F.binary_cross_entropy_with_logits(
                eos_logits, eos_target, reduction="none"
            )
            eos_loss = eos_loss * block_mask.unsqueeze(-1).float()
            eos_loss = eos_loss.sum() / block_mask.sum().clamp(min=1.0)
        else:
            eos_loss = F.binary_cross_entropy_with_logits(
                eos_logits, eos_target, reduction="mean"
            )

        total_loss = (
            self.diff_weight * diff_loss
            + self.clm_weight * token_loss
            + self.eos_weight * eos_loss
        )

        return total_loss, {
            "diff_loss": diff_loss.detach().item(),
            "token_loss": token_loss.detach().item(),
            "eos_loss": eos_loss.detach().item(),
            "total_loss": total_loss.detach().item(),
            "clm_weight": self.clm_weight,
        }


def compute_diffusion_loss(model_output, block_tokens, loss_fn, block_mask=None):
    return loss_fn(model_output, block_tokens, block_mask=block_mask)
