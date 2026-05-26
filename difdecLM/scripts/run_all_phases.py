import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from difdecLM import DifDecConfig
from difdecLM.scripts.phase1_stabilize import phase1
from difdecLM.scripts.phase2_align import phase2
from difdecLM.scripts.phase3_integrate import phase3


def run_pipeline(config_overrides=None, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    config = DifDecConfig()

    print("=" * 60)
    print("  PHASE 1: Stabilize Diffusion Decoder")
    print("  Freeze backbone, train decoder with diffusion loss only")
    print("=" * 60)
    model, trainer1 = phase1(config_overrides)
    ckpt1 = os.path.join(checkpoint_dir, "phase1_final.pt")
    trainer1.save_checkpoint(ckpt1)

    print("\n" + "=" * 60)
    print("  PHASE 2: Align Token Space")
    print("  Add CLM loss, anchor decoder to vocabulary")
    print("=" * 60)
    model, trainer2 = phase2(ckpt1, config_overrides)
    ckpt2 = os.path.join(checkpoint_dir, "phase2_final.pt")
    trainer2.save_checkpoint(ckpt2)

    print("\n" + "=" * 60)
    print("  PHASE 3: Backbone Integration")
    print("  Unfreeze partial backbone, joint training")
    print("=" * 60)
    model, trainer3 = phase3(ckpt2, config_overrides)
    ckpt3 = os.path.join(checkpoint_dir, "phase3_final.pt")
    trainer3.save_checkpoint(ckpt3)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Checkpoints saved to {checkpoint_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
