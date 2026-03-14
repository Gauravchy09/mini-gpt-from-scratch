from __future__ import annotations

from typing import Optional

import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Sample one token from logits of shape [batch_size, vocab_size]."""
    logits = logits / max(temperature, 1e-8)

    if top_k is not None:
        top_k = max(top_k, 1)
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        cutoff = values[:, [-1]]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
