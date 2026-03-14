from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.gpt_model import GPTConfig, MiniGPT


def test_model_forward_shapes() -> None:
    config = GPTConfig(
        vocab_size=50,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        context_length=16,
        dropout=0.0,
    )
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.context_length))
    logits, loss = model(x, x)

    assert logits.shape == (2, config.context_length, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_model_generate_length() -> None:
    config = GPTConfig(
        vocab_size=30,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        context_length=8,
        dropout=0.0,
    )
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=6)
    assert out.shape[1] == 10
