import torch
import torch.nn.functional as F

from ..training.diffusion_process import DiffusionProcess


class BlockGenerator:
    def __init__(self, model, config, device="cuda"):
        self.model = model
        self.config = config
        self.device = device
        self.diff_process = DiffusionProcess(config).to(device)
        self.block_size = config.block.block_size
        self.eos_threshold = config.block.eos_threshold
        self.n_context_slots = config.block.n_context_slots

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.backbone.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, prompt, max_new_blocks=8, temperature=1.0, top_k=0, top_p=0.9, verbose=False):
        self.model.eval()

        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            prompt_ids = prompt.clone().to(self.device)
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0)

        B = prompt_ids.size(0)
        all_ids = prompt_ids.clone()

        for block_idx in range(max_new_blocks):
            context = self._get_context(all_ids, block_idx, verbose)

            block_emb = self.diff_process.sample(
                self.model, context, self.block_size, self.model.d_decoder, verbose
            )

            logits, eos_logits = self.model.projection_head(block_emb)

            eos_probs = torch.sigmoid(eos_logits)

            logits = logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k, dim=-1)
                min_values = values[:, :, -1:]
                logits = torch.where(logits < min_values, float('-inf'), logits)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
                sorted_indices_to_remove[:, :, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    2, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            block_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(B, -1)

            output_len = self.block_size
            for pos in range(self.block_size):
                if eos_probs[0, pos] > self.eos_threshold:
                    output_len = pos + 1
                    break

            block_tokens = block_tokens[:, :output_len]
            all_ids = torch.cat([all_ids, block_tokens], dim=1)

            if verbose:
                generated = self.tokenizer.decode(block_tokens[0], skip_special_tokens=True)
                print(f"  Block {block_idx + 1}/{max_new_blocks}: {generated[:80]}...")

        output = self.tokenizer.decode(all_ids[0], skip_special_tokens=True)
        return output, all_ids

    def _get_context(self, all_ids, block_idx, verbose=False):
        with torch.no_grad():
            backbone_hidden = self.model.backbone(all_ids)
        S = backbone_hidden.size(1)
        ctx_window = self.config.block.context_window
        queries = self.model.context_queries
        scale = self.model.d_backbone ** -0.5

        start = max(0, S - ctx_window)
        window = backbone_hidden[:, start:S, :]
        attn = torch.einsum('sd,btd->bst', queries, window) * scale
        attn = torch.softmax(attn, dim=-1)
        slots = torch.einsum('bst,btd->bsd', attn, window)
        return slots

    def generate_teacherless(self, prompt, max_new_blocks=8, guidance_scale=2.0, verbose=False):
        self.model.eval()

        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        else:
            prompt_ids = prompt.clone().to(self.device)
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0)

        B = prompt_ids.size(0)
        all_ids = prompt_ids.clone()

        for block_idx in range(max_new_blocks):
            cond_context = self._get_context(all_ids, block_idx, verbose)

            uncond_prompt = torch.tensor([[self.tokenizer.eos_token_id] * 1], device=self.device)
            with torch.no_grad():
                uncond_hidden = self.model.backbone(uncond_prompt)
            uncond_h = uncond_hidden[:, -1:, :]
            queries = self.model.context_queries
            scale = self.model.d_backbone ** -0.5
            attn = torch.einsum('sd,btd->bst', queries, uncond_h) * scale
            attn = torch.softmax(attn, dim=-1)
            uncond_ctx = torch.einsum('bst,btd->bsd', attn, uncond_h)

            def guided_model(emb, ctx, t_emb):
                cond_pred = self.model.decoder(emb, ctx, t_emb)
                uncond_pred = self.model.decoder(emb, uncond_ctx, t_emb)
                return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

            block_emb = self._sample_with_fn(
                guided_model, cond_context, block_size=self.block_size,
                d_decoder=self.model.d_decoder, verbose=verbose
            )

            logits, eos_logits = self.model.projection_head(block_emb)
            probs = F.softmax(logits, dim=-1)
            block_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(B, -1)

            eos_probs = torch.sigmoid(eos_logits)
            output_len = self.block_size
            for pos in range(self.block_size):
                if eos_probs[0, pos] > self.eos_threshold:
                    output_len = pos + 1
                    break

            block_tokens = block_tokens[:, :output_len]
            all_ids = torch.cat([all_ids, block_tokens], dim=1)

        output = self.tokenizer.decode(all_ids[0], skip_special_tokens=True)
        return output, all_ids

    def _sample_with_fn(self, denoise_fn, context, block_size, d_decoder, verbose=False):
        K = self.diff_process.config.sampling_steps
        timesteps = self.diff_process.get_ddim_timesteps(context.device)
        ab = self.diff_process.alpha_bar
        device = context.device

        x = torch.randn(context.size(0), block_size, d_decoder, device=device)

        for i in range(K):
            t_val = timesteps[i].item()
            t_batch = torch.full((context.size(0),), t_val, device=device, dtype=torch.float)
            t_emb = self.model.time_embedding(t_batch)

            noise_pred_raw = denoise_fn(x, context, t_emb)

            ab_t = ab[t_val]
            sqrt_ab_t = ab_t.sqrt()
            sqrt_one_minus_ab_t = (1.0 - ab_t).clamp(min=0.0).sqrt()

            noise_pred = self.model.predict_noise(
                noise_pred_raw, x,
                sqrt_alpha_bar=sqrt_ab_t,
                sqrt_one_minus_alpha_bar=sqrt_one_minus_ab_t,
            )

            x0_pred = (x - sqrt_one_minus_ab_t * noise_pred) / sqrt_ab_t.clamp(min=1e-8)

            if i < K - 1:
                ab_next = ab[timesteps[i + 1].item()]
            else:
                ab_next = torch.tensor(1.0, device=device)

            sqrt_ab_next = ab_next.sqrt()
            sqrt_one_minus_ab_next = (1.0 - ab_next).clamp(min=0.0).sqrt()
            x = sqrt_ab_next * x0_pred + sqrt_one_minus_ab_next * noise_pred

        return x
