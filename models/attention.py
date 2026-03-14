from __future__ import annotations

import torch
from torch import nn


class MultiHeadCausalSelfAttention(nn.Module):
    """Readable multi-head causal self-attention implementation."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # One linear layer projects the input into queries, keys, and values.
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split the embedding dimension across attention heads.
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        # The causal mask prevents the model from looking at future tokens.
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Combine the head outputs back into one embedding per token.
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
