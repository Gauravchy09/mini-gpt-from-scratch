from __future__ import annotations

import torch
from torch import nn


class TokenPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        del batch_size  # Not used directly, but left for readability.

        positions = torch.arange(seq_len, device=input_ids.device)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        return self.dropout(token_emb + pos_emb)
