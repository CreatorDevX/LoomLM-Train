"""Quick verification that the architecture is correct."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from difdecLM import DifDecConfig
from difdecLM.model.difdec_lm import DifDecLM

def test_instantiation():
    config = DifDecConfig()
    config.backbone.freeze = True
    config.training.phase = 1

    print(f"  Backbone: {config.backbone.model_name}")
    print(f"  Block size: {config.block.block_size}")
    print(f"  Decoder d_model: {config.decoder.d_decoder}")
    print(f"  Decoder layers: {config.decoder.n_layers}")
    print(f"  Decoder heads: {config.decoder.n_heads}")
    print(f"  Diffusion timesteps: {config.diffusion.timesteps}")
    print(f"  Sampling steps: {config.diffusion.sampling_steps}")

    model = DifDecLM(config)
    total = model.get_num_params()
    trainable = model.get_num_trainable()

    print(f"\n  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params: {total - trainable:,}")

    for name, param in model.named_parameters():
        size = param.numel()
        if param.requires_grad:
            print(f"    TRAINABLE: {name}: {size:,}")

    backbone_total = sum(p.numel() for p in model.backbone.parameters())
    backbone_trainable = model.backbone.get_num_trainable()
    decoder_total = model.get_num_params() - backbone_total

    print(f"\n  Backbone total: {backbone_total:,}")
    print(f"  Backbone frozen: {backbone_total - backbone_trainable:,}")
    print(f"  Decoder + head total: {decoder_total:,}")
    print(f"  Decoder + head trainable: {trainable - backbone_trainable:,}")

    # Count decoder-only params
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    time_params = sum(p.numel() for p in model.time_embedding.parameters())
    proj_params = sum(p.numel() for p in model.projection_head.parameters())
    embed_proj_params = sum(p.numel() for p in model.embed_proj.parameters()) if hasattr(model, 'embed_proj') else 0
    print(f"\n  decoder stack: {decoder_params:,}")
    print(f"  time embedding: {time_params:,}")
    print(f"  projection head: {proj_params:,}")
    print(f"  embed projection: {embed_proj_params:,}")
    print(f"  total decoder head: {decoder_params + time_params + proj_params + embed_proj_params:,}")

    return model


def test_forward_pass(model):
    config = model.config
    B = 2
    n_blocks = 3
    block_size = config.block.block_size
    seq_len = n_blocks * block_size
    S = seq_len  # just blocks, no special prompt

    input_ids = torch.randint(0, 1000, (B, S))
    timesteps = torch.randint(0, config.diffusion.timesteps, (B, n_blocks))

    print(f"\n  Input shape: {input_ids.shape}")
    print(f"  Timesteps shape: {timesteps.shape}")

    with torch.no_grad():
        output = model(input_ids, timesteps)

    print(f"  noise_pred shape: {output['noise_pred'].shape}")
    print(f"  noise_target shape: {output['noise_target'].shape}")
    print(f"  logits shape: {output['logits'].shape}")
    print(f"  eos_logits shape: {output['eos_logits'].shape}")
    print(f"  clean_embeddings shape: {output['clean_embeddings'].shape}")

    # Verify shapes
    expected_noise_shape = (B, n_blocks, block_size, config.decoder.d_decoder)
    expected_logit_shape = (B, n_blocks, block_size, config.vocab_size)
    expected_eos_shape = (B, n_blocks, block_size)

    assert output['noise_pred'].shape == expected_noise_shape, \
        f"noise_pred shape mismatch: {output['noise_pred'].shape} != {expected_noise_shape}"
    assert output['logits'].shape == expected_logit_shape, \
        f"logits shape mismatch: {output['logits'].shape} != {expected_logit_shape}"
    assert output['eos_logits'].shape == expected_eos_shape, \
        f"eos_logits shape mismatch: {output['eos_logits'].shape} != {expected_eos_shape}"

    # Verify loss computation
    from difdecLM.training.losses import DiffusionLoss
    loss_fn = DiffusionLoss(config)
    block_tokens = input_ids.view(B, n_blocks, block_size)
    total_loss, metrics = loss_fn(output, block_tokens)
    print(f"\n  Total loss: {total_loss.item():.4f}")
    print(f"  Diffusion loss: {metrics['diff_loss']:.4f}")
    print(f"  Token loss: {metrics['token_loss']:.4f}")
    print(f"  EOS loss: {metrics['eos_loss']:.4f}")

    total_loss.backward()
    grad_norms = []
    for p in model.get_trainable_params():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print(f"  Gradient norms: max={max(grad_norms):.4f}, min={min(grad_norms):.4f}, nonzero={len(grad_norms)}")

    # Verify gradient flows to decoder
    decoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.decoder.named_parameters()
    )
    print(f"  Decoder has gradient: {decoder_has_grad}")

    print("\n  All forward pass tests PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("  DifDecLM Architecture Test")
    print("=" * 60)

    print("\n  Instantiating model...")
    model = test_instantiation()

    print("\n" + "-" * 60)
    print("  Testing forward pass...")
    test_forward_pass(model)

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)
