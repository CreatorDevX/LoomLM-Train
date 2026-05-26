from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BackboneConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    d_backbone: int = 576
    freeze: bool = True
    unfreeze_last_n_layers: int = 0
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16


@dataclass
class DecoderConfig:
    d_decoder: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    activation: str = "silu"
    norm_eps: float = 1e-5

    conditioning: str = "film"
    d_context: int = 576

    embedding_mode: str = "shared"
    use_weight_tying: bool = True


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    sampling_steps: int = 8
    noise_schedule: str = "cosine"
    prediction_type: str = "epsilon"

    beta_start: float = 1e-4
    beta_end: float = 0.02
    cosine_s: float = 0.008

    d_time_embed: int = 384
    time_embed_mlp: bool = True


@dataclass
class BlockConfig:
    block_size: int = 64
    max_blocks: int = 32
    max_seq_len: int = 2048

    context_pooling: str = "last"  # "last", "mean", "learned_query"
    context_window: int = 1

    eos_threshold: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    warmup_steps: int = 1000

    lr: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    diffusion_loss_weight: float = 1.0
    clm_loss_weight: float = 0.0
    eos_loss_weight: float = 0.01
    context_loss_weight: float = 0.0

    clm_loss_ramp_steps: int = 5000
    clm_loss_max_weight: float = 1.0

    mask_pad_positions: bool = True
    pad_token_id: int = 0

    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "checkpoints"

    phase: int = 1


@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "default"
    split: str = "train"
    text_field: str = "text"
    max_samples: Optional[int] = 100000
    cache_dir: Optional[str] = None
    num_workers: int = 0
    shuffle: bool = True
    streaming: bool = True
    use_random_fallback: bool = False


@dataclass
class InferenceConfig:
    max_new_blocks: int = 8
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    use_ema: bool = False
    verbose: bool = False


@dataclass
class DifDecConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    seed: int = 42
    device: str = "cuda"
    dtype: str = "float32"
    compile: bool = False

    @property
    def vocab_size(self) -> int:
        return 49152

    @property
    def total_max_seq_len(self) -> int:
        return self.block.max_blocks * self.block.block_size

    def to_dict(self) -> dict:
        return {
            "backbone": self.backbone.__dict__,
            "decoder": self.decoder.__dict__,
            "diffusion": self.diffusion.__dict__,
            "block": self.block.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "inference": self.inference.__dict__,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
            "compile": self.compile,
        }
