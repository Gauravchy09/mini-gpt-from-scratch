from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tokenizer.vocab import Vocabulary


class SimpleCharTokenizer:
    """Character-level tokenizer aimed at clarity over complexity."""

    def __init__(self, vocab: Vocabulary | None = None) -> None:
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        if self.vocab is None:
            raise ValueError("Tokenizer is not fitted yet. Call fit() first.")
        return self.vocab.size

    def fit(self, text: str) -> None:
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text")
        # This project keeps tokenization character-level so the mapping is easy to explain.
        self.vocab = Vocabulary.from_text(text)

    def encode(self, text: str) -> List[int]:
        self._ensure_fitted()
        return self.vocab.encode(text)

    def decode(self, token_ids: Iterable[int]) -> str:
        self._ensure_fitted()
        return self.vocab.decode(token_ids)

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        self.vocab.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleCharTokenizer":
        vocab = Vocabulary.load(path)
        return cls(vocab=vocab)

    def _ensure_fitted(self) -> None:
        if self.vocab is None:
            raise ValueError("Tokenizer is not fitted yet. Call fit() first.")
