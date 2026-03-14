from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """Simple character-token dataset for next-token prediction."""

    def __init__(self, token_ids: List[int], block_size: int) -> None:
        if len(token_ids) <= block_size:
            raise ValueError("token_ids length must be greater than block_size")
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # `x` is the current context and `y` is the same sequence shifted by one token.
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_text(data_path: str | Path) -> str:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_train_val_datasets(
    token_ids: List[int],
    block_size: int,
    train_split: float = 0.9,
) -> Tuple[NextTokenDataset, NextTokenDataset]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    # Keep validation usable even for very small demo datasets.
    if len(train_ids) <= block_size:
        train_ids = token_ids
    if len(val_ids) <= block_size:
        val_ids = token_ids[-(block_size + 1) :]

    train_dataset = NextTokenDataset(train_ids, block_size)
    val_dataset = NextTokenDataset(val_ids, block_size)
    return train_dataset, val_dataset
