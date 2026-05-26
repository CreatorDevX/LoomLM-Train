import torch
from difdecLM import DifDecConfig
from difdecLM.model.difdec_lm import DifDecLM
from difdecLM.data import BlockDiffusionDataset, create_dataloader
from difdecLM.training import Trainer


def phase2(checkpoint_path=None, config_overrides=None):
    config = DifDecConfig()

    if config_overrides:
        for k, v in config_overrides.items():
            parts = k.split(".")
            obj = config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)

    config.training.phase = 2
    config.backbone.freeze = True
    config.backbone.unfreeze_last_n_layers = 0
    config.training.diffusion_loss_weight = 1.0
    config.training.clm_loss_weight = 0.3
    config.training.eos_loss_weight = 0.05
    config.training.clm_loss_ramp_steps = 3000
    config.training.clm_loss_max_weight = 1.0
    config.training.lr = 5e-5
    config.training.max_steps = 80000

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = DifDecLM(config).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded checkpoint from {checkpoint_path}")

    print(f"  Trainable params: {model.get_num_trainable():,}")
    print(f"  Total params: {model.get_num_params():,}")

    dataset = BlockDiffusionDataset(config)
    dataloader = create_dataloader(dataset, config)

    trainer = Trainer(model, config, device=device)
    trainer.train(dataloader)

    return model, trainer


if __name__ == "__main__":
    phase2()
