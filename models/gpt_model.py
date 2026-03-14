from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.embedding import TokenPositionEmbedding
from models.transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    context_length: int
    dropout: float = 0.1


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = TokenPositionEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            context_length=config.context_length,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    context_length=config.context_length,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("Input sequence is longer than context_length")

        # Convert token ids into contextual token representations.
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            # Only the latest context window is fed back into the model.
            context = input_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                top_k = max(top_k, 1)
                values, _ = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                cutoff = values[:, [-1]]
                next_token_logits = torch.where(
                    next_token_logits < cutoff,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
