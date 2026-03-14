from __future__ import annotations

import torch
import torch.nn.functional as F


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for next-token prediction."""
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
