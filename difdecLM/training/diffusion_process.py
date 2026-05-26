import torch
import torch.nn.functional as F
import math


class DiffusionProcess:
    def __init__(self, config):
        self.config = config.diffusion
        self.blk = config.block
        self.T = self.config.timesteps
        self.device = None

        self._precompute_schedule()

    def _precompute_schedule(self):
        T = self.T
        if self.config.noise_schedule == "cosine":
            s = self.config.cosine_s
            t = torch.arange(T + 1).float() / T
            f = torch.cos((t + s) / (1.0 + s) * (math.pi / 2.0))
            alpha_bar = f.clamp(min=0.0, max=1.0)
        else:
            beta = torch.linspace(self.config.beta_start, self.config.beta_end, T)
            alpha = 1.0 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
            alpha_bar = torch.cat([torch.ones(1), alpha_bar])

        self.register("alpha_bar", alpha_bar)

    def register(self, name, tensor):
        tensor = tensor.clone()
        if self.device is not None:
            tensor = tensor.to(self.device)
        setattr(self, f"_{name}", tensor)

    def to(self, device):
        self.device = device
        for name in ["alpha_bar"]:
            if hasattr(self, f"_{name}"):
                setattr(self, f"_{name}", getattr(self, f"_{name}").to(device))
        return self

    @property
    def alpha_bar(self):
        if not hasattr(self, "_alpha_bar"):
            self._precompute_schedule()
            if self.device is not None:
                self._alpha_bar = self._alpha_bar.to(self.device)
        return self._alpha_bar

    def q_sample(self, x0, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar = self.alpha_bar[timesteps.long()]

        while alpha_bar.dim() < x0.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        shape = list(x0.shape)
        alpha_bar = alpha_bar.expand(shape)

        sqrt_ab = alpha_bar.sqrt()
        sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar.clamp(max=1.0))

        return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise

    def get_timesteps(self, batch_size, n_blocks, device):
        t = torch.randint(0, self.T, (batch_size, n_blocks), device=device)
        return t

    def get_ddim_timesteps(self, device):
        K = self.config.sampling_steps
        T = self.T
        steps = torch.linspace(T - 1, 0, K, device=device).long()
        return steps

    @torch.no_grad()
    def ddim_sample(self, model, context, block_size, d_decoder, verbose=False):
        K = self.config.sampling_steps
        timesteps = self.get_ddim_timesteps(context.device)
        ab = self.alpha_bar
        B = context.size(0)
        device = context.device

        x = torch.randn(B, block_size, d_decoder, device=device)
        time_emb_fn = model.time_embedding

        for i in range(K):
            t_val = timesteps[i].item()
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.float)
            t_emb = time_emb_fn(t_batch)

            noise_pred_raw = model.decoder(x, context, t_emb)

            ab_t = ab[t_val]
            sqrt_ab_t = ab_t.sqrt()
            sqrt_one_minus_ab_t = (1.0 - ab_t).clamp(min=0.0).sqrt()

            noise_pred = model.predict_noise(
                noise_pred_raw, x,
                sqrt_alpha_bar=sqrt_ab_t,
                sqrt_one_minus_alpha_bar=sqrt_one_minus_ab_t,
            )

            x0_pred = (x - sqrt_one_minus_ab_t * noise_pred) / sqrt_ab_t.clamp(min=1e-8)

            if i < K - 1:
                t_next_val = timesteps[i + 1].item()
                ab_next = ab[t_next_val]
            else:
                ab_next = torch.tensor(1.0, device=device)

            sqrt_ab_next = ab_next.sqrt()
            sqrt_one_minus_ab_next = (1.0 - ab_next).clamp(min=0.0).sqrt()

            x = sqrt_ab_next * x0_pred + sqrt_one_minus_ab_next * noise_pred

            if verbose:
                print(f"  DDIM step {i + 1}/{K}: t={t_val}, noise_std={noise_pred.std().item():.4f}")

        return x

    @torch.no_grad()
    def sample(self, model, context, block_size, d_decoder, verbose=False):
        return self.ddim_sample(model, context, block_size, d_decoder, verbose)
