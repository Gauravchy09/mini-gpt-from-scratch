from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from models.gpt_model import GPTConfig


@dataclass
class NotebookModelConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout: float


def build_model_config(config_path: str | Path) -> GPTConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    cfg = NotebookModelConfig(**payload)
    return GPTConfig(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
