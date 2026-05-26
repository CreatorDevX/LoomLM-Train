from .config import DifDecConfig

def create_model(config):
    from .model.difdec_lm import DifDecLM
    return DifDecLM(config)

__all__ = ["DifDecConfig", "create_model"]
