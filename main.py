import os
import sys
import argparse
import torch

from difdecLM import DifDecConfig
from difdecLM.model.difdec_lm import DifDecLM
from difdecLM.training import Trainer
from difdecLM.data import BlockDiffusionDataset, create_dataloader
from difdecLM.inference import BlockGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="DifDecLM: Block Diffusion Language Model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate", "interactive"])
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3], help="Training phase")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Generation prompt")
    parser.add_argument("--max_blocks", type=int, default=None, help="Max blocks to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_config(args):
    config = DifDecConfig()
    config.device = args.device
    config.training.phase = args.phase
    config.training.output_dir = args.output_dir

    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.lr:
        config.training.lr = args.lr
    if args.seed:
        config.seed = args.seed

    if args.phase == 1:
        config.backbone.freeze = True
        config.backbone.unfreeze_last_n_layers = 0
        config.training.diffusion_loss_weight = 1.0
        config.training.clm_loss_weight = 0.0
        config.training.eos_loss_weight = 0.01
        config.training.lr = args.lr or 1e-4
        config.training.max_steps = args.max_steps or 50000
    elif args.phase == 2:
        config.backbone.freeze = True
        config.backbone.unfreeze_last_n_layers = 0
        config.training.diffusion_loss_weight = 1.0
        config.training.clm_loss_weight = 0.3
        config.training.eos_loss_weight = 0.05
        config.training.clm_loss_ramp_steps = 3000
        config.training.clm_loss_max_weight = 1.0
        config.training.lr = args.lr or 5e-5
        config.training.max_steps = args.max_steps or 80000
    elif args.phase == 3:
        config.backbone.freeze = False
        config.backbone.unfreeze_last_n_layers = 4
        config.training.diffusion_loss_weight = 0.5
        config.training.clm_loss_weight = 0.5
        config.training.eos_loss_weight = 0.02
        config.training.clm_loss_ramp_steps = 2000
        config.training.clm_loss_max_weight = 1.0
        config.training.context_loss_weight = 0.1
        config.training.lr = args.lr or 2e-5
        config.training.max_steps = args.max_steps or 120000

    return config


def train(args):
    config = setup_config(args)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    print(f"  Device: {device}")
    print(f"  Phase: {config.training.phase}")
    print(f"  Output dir: {config.training.output_dir}")

    model = DifDecLM(config).to(device)
    print(f"  Trainable params: {model.get_num_trainable():,}")
    print(f"  Total params: {model.get_num_params():,}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    print("  Loading dataset...")
    dataset = BlockDiffusionDataset(config)
    dataloader = create_dataloader(dataset, config)

    trainer = Trainer(model, config, device=device)
    trainer.train(dataloader)

    model.eval()
    test_prompts = [
        "The future of AI is",
        "In the beginning,",
        "The key to understanding",
    ]
    generator = BlockGenerator(model, config, device=device)
    for prompt in test_prompts:
        output, _ = generator.generate(
            prompt,
            max_new_blocks=2,
            temperature=0.8,
            top_p=0.9,
            verbose=False,
        )
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {output[:200]}...")


def generate(args):
    config = DifDecConfig()
    config.device = args.device

    if not args.checkpoint:
        print("  Error: --checkpoint required for generation mode")
        return

    device = torch.device(config.device)
    model = DifDecLM(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    generator = BlockGenerator(model, config, device=device)
    max_blocks = args.max_blocks or config.inference.max_new_blocks

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Once upon a time,",
            "The meaning of life is",
            "In recent years,",
        ]

    for prompt in prompts:
        output, tokens = generator.generate(
            prompt,
            max_new_blocks=max_blocks,
            temperature=args.temperature,
            top_p=0.9,
            verbose=True,
        )
        print(f"\n  Prompt: {prompt}")
        print(f"  Generated ({tokens.size(1)} tokens):")
        print(f"  {output}")
        print("-" * 60)


def interactive(args):
    config = DifDecConfig()
    config.device = args.device

    if not args.checkpoint:
        print("  Error: --checkpoint required for interactive mode")
        return

    device = torch.device(config.device)
    model = DifDecLM(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    generator = BlockGenerator(model, config, device=device)
    max_blocks = args.max_blocks or 4

    print("\n  Interactive mode. Enter prompts (Ctrl+C to exit).\n")
    try:
        while True:
            prompt = input("  Prompt> ").strip()
            if not prompt:
                continue
            output, _ = generator.generate(
                prompt,
                max_new_blocks=max_blocks,
                temperature=args.temperature,
                top_p=0.9,
                verbose=False,
            )
            print(f"  Output: {output}\n")
    except KeyboardInterrupt:
        print("\n  Goodbye!")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
    elif args.mode == "interactive":
        interactive(args)


if __name__ == "__main__":
    main()
