import torch
from difdecLM import DifDecConfig
from difdecLM.model.difdec_lm import DifDecLM
from difdecLM.data import BlockDiffusionDataset, create_dataloader
from difdecLM.training import Trainer


def phase1(config_overrides=None):
    config = DifDecConfig()

    if config_overrides:
        for k, v in config_overrides.items():
            parts = k.split(".")
            obj = config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], v)

    config.training.phase = 1
    config.backbone.freeze = True
    config.backbone.unfreeze_last_n_layers = 0
    config.training.diffusion_loss_weight = 1.0
    config.training.clm_loss_weight = 0.0
    config.training.eos_loss_weight = 0.01
    config.training.clm_loss_ramp_steps = 0
    config.training.lr = 1e-4
    config.training.max_steps = 50000

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = DifDecLM(config).to(device)
    print(f"  Trainable params: {model.get_num_trainable():,}")
    print(f"  Total params: {model.get_num_params():,}")

    dataset = BlockDiffusionDataset(config)
    dataloader = create_dataloader(dataset, config)

    trainer = Trainer(model, config, device=device)
    trainer.train(dataloader)

    return model, trainer


if __name__ == "__main__":
    phase1()
