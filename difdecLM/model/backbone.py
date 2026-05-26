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
        x = self.embed_fn(input_ids)

        if hasattr(self.backbone, 'drop'):
            x = self.backbone.drop(x)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)[0]

        x = self.norm(x)
        return x

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
