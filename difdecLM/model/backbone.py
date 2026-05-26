import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def _str_to_dtype(s: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(s, torch.float32)


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=16, alpha=16):
        super().__init__()
        self.add_module("original", original_linear)
        self.r = r
        self.scaling = alpha / r
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(original_linear.in_features, r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, original_linear.out_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        original_linear.requires_grad_(False)

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAEmbedding(nn.Module):
    def __init__(self, original_embedding, r=16, alpha=16):
        super().__init__()
        self.add_module("original", original_embedding)
        self.r = r
        self.scaling = alpha / r
        d_model = original_embedding.embedding_dim
        device = original_embedding.weight.device
        dtype = original_embedding.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(d_model, r, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, d_model, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        original_embedding.requires_grad_(False)

    def forward(self, x):
        h = self.original(x)
        return h + (h @ self.lora_A @ self.lora_B) * self.scaling


def _detect_model(model):
    if hasattr(model, 'transformer'):
        backbone = model.transformer
        if hasattr(backbone, 'wte'):
            embed_fn = backbone.wte
        elif hasattr(backbone, 'embed_tokens'):
            embed_fn = backbone.embed_tokens
        else:
            embed_fn = model.get_input_embeddings()
        if hasattr(backbone, 'h'):
            layers = backbone.h
        elif hasattr(backbone, 'layers'):
            layers = backbone.layers
        else:
            layers = backbone
        if hasattr(backbone, 'ln_f'):
            norm = backbone.ln_f
        elif hasattr(backbone, 'norm'):
            norm = backbone.norm
        else:
            norm = nn.Identity()
    elif hasattr(model, 'model'):
        backbone = model.model
        embed_fn = backbone.embed_tokens if hasattr(backbone, 'embed_tokens') else model.get_input_embeddings()
        layers = backbone.layers if hasattr(backbone, 'layers') else backbone
        norm = backbone.norm if hasattr(backbone, 'norm') else nn.Identity()
    else:
        raise ValueError(f"Unsupported model architecture: {type(model).__name__}")

    return backbone, embed_fn, layers, norm


class SmolLM2Backbone(nn.Module):
    def __init__(self, config, dtype="float32"):
        super().__init__()
        self.config = config
        self.d_backbone = config.d_backbone

        hf_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=_str_to_dtype(dtype),
            trust_remote_code=True,
        )

        self.backbone, self.embed_fn, self.layers, self.norm = _detect_model(hf_model)
        self.lm_head = None

        if config.freeze:
            self._freeze_all()

        self._maybe_unfreeze_layers(config.unfreeze_last_n_layers)

        self.register_buffer(
            "embedding_weight",
            self.embed_fn.weight.data.detach().clone(),
            persistent=False,
        )

    def _freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _maybe_unfreeze_layers(self, n):
        if n <= 0:
            return
        if hasattr(self.layers, 'children'):
            layer_list = list(self.layers.children())
        else:
            layer_list = list(self.layers)
        for layer in layer_list[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)

    def get_embedding_matrix(self):
        return self.embed_fn.weight

    def embed_tokens(self, input_ids):
        return self.embed_fn(input_ids)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def apply_lora(self, r=16, alpha=16):
        def _recurse(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    lora = LoRALinear(child, r, alpha)
                    module._modules[name] = lora
                else:
                    _recurse(child)
        _recurse(self.backbone)
        lora_emb = LoRAEmbedding(self.embed_fn, r, alpha)
        self.embed_fn = lora_emb

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
