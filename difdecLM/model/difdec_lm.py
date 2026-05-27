import torch
import torch.nn as nn

from .backbone import SmolLM2Backbone
from .time_embedding import TimeEmbedding
from .diffusion_decoder import DiffusionDecoderStack
from .projection_head import TokenProjectionHead, EmbeddingProjectionHead


class DifDecLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dc = config.decoder
        bc = config.backbone
        difc = config.diffusion
        blc = config.block
        v = config.vocab_size

        self.block_size = blc.block_size
        self.d_backbone = bc.d_backbone
        self.d_decoder = dc.d_decoder

        self.backbone = SmolLM2Backbone(bc, config.dtype)

        self.time_embedding = TimeEmbedding(difc.d_time_embed, use_mlp=difc.time_embed_mlp)
        self.decoder = DiffusionDecoderStack(config)

        if dc.embedding_mode == "shared":
            self.embed_proj = nn.Linear(self.d_backbone, self.d_decoder)
            self.projection_head = EmbeddingProjectionHead(
                d_decoder=dc.d_decoder,
                d_backbone=bc.d_backbone,
                vocab_size=v,
                embedding_weight=self.backbone.get_embedding_matrix(),
            )
        else:
            self.token_embed = nn.Embedding(v, self.d_decoder)
            embed_weight = self.token_embed.weight if dc.use_weight_tying else None
            self.projection_head = TokenProjectionHead(
                d_decoder=dc.d_decoder,
                vocab_size=v,
                use_weight_tying=dc.use_weight_tying,
                embedding_weight=embed_weight,
            )

        self.n_context_slots = blc.n_context_slots
        self.context_queries = nn.Parameter(
            torch.randn(self.n_context_slots, self.d_backbone) * 0.02
        )

    def get_clean_embeddings(self, block_tokens):
        if hasattr(self, 'token_embed'):
            return self.token_embed(block_tokens)
        backbone_emb = self.backbone.embed_tokens(block_tokens)
        return self.embed_proj(backbone_emb)

    def prepare_context(self, backbone_hidden, n_blocks):
        B, S, D = backbone_hidden.shape
        block_size = self.block_size
        n_slots = self.n_context_slots
        ctx_window = self.config.block.context_window

        queries = self.context_queries
        scale = D ** -0.5

        ctx = torch.zeros(B, n_blocks, n_slots, D, device=backbone_hidden.device)

        for b in range(n_blocks):
            pos = b * block_size
            if pos == 0:
                window = backbone_hidden[:, 0:1, :]
            else:
                start = max(0, pos - ctx_window)
                window = backbone_hidden[:, start:pos, :]
            attn = torch.einsum('sd,btd->bst', queries, window) * scale
            attn = torch.softmax(attn, dim=-1)
            slots = torch.einsum('bst,btd->bsd', attn, window)
            ctx[:, b] = slots

        return ctx

    def predict_noise(self, noise_pred, E_t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
        ptype = self.config.diffusion.prediction_type
        if ptype == "epsilon":
            return noise_pred
        elif ptype == "x0":
            x0_pred = noise_pred
            eps = sqrt_one_minus_alpha_bar.clamp(min=1e-8)
            return (E_t - sqrt_alpha_bar * x0_pred) / eps
        elif ptype == "v":
            v_pred = noise_pred
            return sqrt_alpha_bar * v_pred + sqrt_one_minus_alpha_bar * E_t
        else:
            return noise_pred

    def forward(self, input_ids, timesteps, noise=None, attention_mask=None):
        B, S = input_ids.shape
        block_size = self.block_size
        n_blocks = S // block_size

        assert n_blocks > 0, "Sequence must contain at least one block"
        assert timesteps.shape == (B, n_blocks), f"timesteps shape {timesteps.shape} != ({B}, {n_blocks})"

        backbone_hidden = self.backbone(input_ids, attention_mask=attention_mask)

        block_tokens = input_ids.view(B, n_blocks, block_size)
        context = self.prepare_context(backbone_hidden, n_blocks)

        E0 = self.get_clean_embeddings(block_tokens)

        if noise is None:
            noise = torch.randn_like(E0)

        sqrt_alpha_bar = self._get_sqrt_alpha_bar(timesteps)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - sqrt_alpha_bar.pow(2).clamp(min=0.0, max=1.0))

        sd = sqrt_alpha_bar.shape
        sqrt_alpha_bar = sqrt_alpha_bar.view(sd[0], sd[1], 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(sd[0], sd[1], 1, 1)

        E_t = sqrt_alpha_bar * E0 + sqrt_one_minus_alpha_bar * noise

        E_t_flat = E_t.view(B * n_blocks, block_size, self.d_decoder)
        n_slots = self.n_context_slots
        context_flat = context.view(B * n_blocks, n_slots, self.d_backbone)

        t_flat = timesteps.view(B * n_blocks).float()
        time_emb = self.time_embedding(t_flat)

        pred = self.decoder(E_t_flat, context_flat, time_emb)

        pred = pred.view(B, n_blocks, block_size, self.d_decoder)

        noise_pred = self.predict_noise(pred, E_t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar)

        logits, eos_logits = self.projection_head(pred)

        return {
            "noise_pred": noise_pred,
            "noise_target": noise,
            "pred_embeddings": pred,
            "clean_embeddings": E0,
            "noisy_embeddings": E_t,
            "logits": logits,
            "eos_logits": eos_logits,
            "context": context,
            "timesteps": timesteps,
            "sqrt_alpha_bar": sqrt_alpha_bar,
            "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        }

    @torch.no_grad()
    def _get_sqrt_alpha_bar(self, timesteps):
        config = self.config.diffusion
        device = timesteps.device
        T = config.timesteps

        if not hasattr(self, '_cached_sqrt_alpha_bar') or self._cached_sqrt_alpha_bar.device != device:
            if config.noise_schedule == "cosine":
                s = config.cosine_s
                t = torch.arange(T + 1, device=device).float() / T
                f = torch.cos((t + s) / (1.0 + s) * (torch.pi / 2.0))
                alpha_bar = f.clamp(min=0.0, max=1.0)
                sqrt_alpha_bar = alpha_bar.sqrt()
            else:
                beta = torch.linspace(config.beta_start, config.beta_end, T, device=device)
                alpha = 1.0 - beta
                alpha_bar = torch.cumprod(alpha, dim=0)
                sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp(min=0.0))
                sqrt_alpha_bar = torch.cat([torch.ones(1, device=device), sqrt_alpha_bar])

            self.register_buffer('_cached_sqrt_alpha_bar', sqrt_alpha_bar, persistent=False)

        return self._cached_sqrt_alpha_bar[timesteps.long()]

    def get_trainable_params(self):
        params = list(self.decoder.parameters())
        params.extend(self.time_embedding.parameters())
        params.extend(self.projection_head.parameters())
        params.append(self.context_queries)
        if hasattr(self, 'embed_proj'):
            params.extend(self.embed_proj.parameters())
        if hasattr(self, 'token_embed'):
            params.extend(self.token_embed.parameters())
        params.extend(self.backbone.get_trainable_params())
        return params

    def get_num_trainable(self):
        return sum(p.numel() for p in self.get_trainable_params())

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
