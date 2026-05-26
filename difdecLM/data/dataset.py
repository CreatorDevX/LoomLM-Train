import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import itertools


class BlockDiffusionDataset(IterableDataset):
    """Streaming dataset from HuggingFace with block-structured tokenization.

    Yields dicts with:
        input_ids:    [seq_len]          tokens[:-1], length = n_blocks * block_size
        block_tokens: [n_blocks, 64]     tokens[1:],  reshaped
        attention_mask: [seq_len]        1 for real, 0 for pad
    """

    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        dc = config.data
        self.block_size = config.block.block_size
        self.max_blocks = config.block.max_blocks
        self.max_seq_len = self.block_size * self.max_blocks
        self.pad_id = config.training.pad_token_id
        self.text_field = dc.text_field
        self.max_samples = dc.max_samples
        self.use_random = dc.use_random_fallback

        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.backbone.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self._iterator = None

    def _stream_hf_dataset(self):
        from datasets import load_dataset
        dc = self.config.data
        ds = load_dataset(
            dc.dataset_name,
            dc.dataset_config,
            split=dc.split,
            streaming=dc.streaming,
            cache_dir=dc.cache_dir,
        )
        if self.max_samples is not None:
            ds = ds.take(self.max_samples)
        for item in ds:
            yield item[self.text_field]

    def _stream_random_texts(self, seed=42):
        import random
        rng = random.Random(seed)
        words = ["the", "a", "an", "in", "of", "to", "and", "is", "it", "that",
                 "intelligence", "learning", "model", "diffusion", "language",
                 "transformer", "attention", "token", "embedding", "block",
                 "neural", "network", "deep", "training", "inference"]
        count = 0
        while self.max_samples is None or count < self.max_samples:
            length = rng.randint(200, self.max_seq_len)
            text = " ".join(rng.choice(words) for _ in range(length))
            yield text
            count += 1

    def _process_text(self, text):
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_seq_len)

        if len(tokens) < self.block_size + 1:
            needed = self.block_size + 1 - len(tokens)
            tokens = tokens + [self.pad_id] * needed

        total_len = ((len(tokens) - 1) // self.block_size) * self.block_size
        if total_len < self.block_size:
            total_len = self.block_size
        tokens = tokens[:total_len + 1]

        input_ids = tokens[:-1]
        block_tokens = tokens[1:]

        n_blocks = total_len // self.block_size
        input_ids = input_ids[:n_blocks * self.block_size]
        block_tokens = block_tokens[:n_blocks * self.block_size]

        attention_mask = [1] * len(input_ids)

        block_tokens = torch.tensor(block_tokens, dtype=torch.long).view(-1, self.block_size)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "block_tokens": block_tokens,
        }

    def __iter__(self):
        source = self._stream_random_texts if self.use_random else self._stream_hf_dataset
        self._iterator = map(self._process_text, source())
        return self._iterator

    def __len__(self):
        if self.max_samples is not None:
            return self.max_samples
        raise TypeError("Streaming dataset has no __len__")


def collate_blocks(batch):
    input_ids_list = []
    attn_mask_list = []
    block_tokens_list = []

    for item in batch:
        input_ids_list.append(item["input_ids"])
        attn_mask_list.append(item["attention_mask"])
        block_tokens_list.append(item["block_tokens"])

    max_len = max(t.shape[0] for t in input_ids_list)
    max_blocks = max(t.shape[0] for t in block_tokens_list)
    bs = len(batch)

    padded_inputs = torch.zeros(bs, max_len, dtype=torch.long)
    padded_mask = torch.zeros(bs, max_len, dtype=torch.long)
    padded_blocks = torch.zeros(bs, max_blocks, block_tokens_list[0].shape[1], dtype=torch.long)
    block_counts = torch.zeros(bs, dtype=torch.long)
    block_mask = torch.zeros(bs, max_blocks, dtype=torch.bool)

    for i, (inp, attn, blk) in enumerate(
        zip(input_ids_list, attn_mask_list, block_tokens_list)
    ):
        seq_len = inp.shape[0]
        n_blk = blk.shape[0]
        padded_inputs[i, :seq_len] = inp
        padded_mask[i, :seq_len] = attn
        padded_blocks[i, :n_blk] = blk
        block_counts[i] = n_blk
        block_mask[i, :n_blk] = True

    return {
        "input_ids": padded_inputs,
        "attention_mask": padded_mask,
        "block_tokens": padded_blocks,
        "block_counts": block_counts,
        "block_mask": block_mask,
    }
