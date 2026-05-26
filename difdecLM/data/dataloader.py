from torch.utils.data import DataLoader, IterableDataset
from .dataset import collate_blocks


def create_dataloader(dataset, config):
    """Create a DataLoader appropriate for the dataset type (streaming or map)."""
    dc = config.data
    tc = config.training

    is_streaming = isinstance(dataset, IterableDataset)

    return DataLoader(
        dataset,
        batch_size=tc.batch_size,
        shuffle=False if is_streaming else dc.shuffle,
        collate_fn=collate_blocks,
        num_workers=0 if is_streaming else dc.num_workers,
        pin_memory=True,
    )
