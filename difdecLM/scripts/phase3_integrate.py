import torch
from difdecLM import DifDecConfig
from difdecLM.model.difdec_lm import DifDecLM
from difdecLM.data import BlockDiffusionDataset, create_dataloader
from difdecLM.training import Trainer


def phase3(checkpoint_path=None, config_overrides=None):
    config = DifDecConfig()

    if config_overrides:
        for k, v in config_overrides.items():
            parts = k.split(".")
            obj = config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)

    config.training.phase = 3
    config.backbone.freeze = False
    config.backbone.unfreeze_last_n_layers = 4
    config.training.diffusion_loss_weight = 0.5
    config.training.clm_loss_weight = 0.5
    config.training.eos_loss_weight = 0.02
    config.training.clm_loss_ramp_steps = 2000
    config.training.clm_loss_max_weight = 1.0
    config.training.context_loss_weight = 0.1
    config.training.lr = 2e-5
    config.training.max_steps = 120000

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
    phase3()
