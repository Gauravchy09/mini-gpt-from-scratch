from __future__ import annotations

import torch
from torch import nn

from models.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim=embedding_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize first, then add the residual connection.
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
