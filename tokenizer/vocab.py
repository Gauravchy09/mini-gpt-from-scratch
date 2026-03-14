from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class Vocabulary:
    tokens: List[str]

    def __post_init__(self) -> None:
        self.stoi: Dict[str, int] = {token: idx for idx, token in enumerate(self.tokens)}
        self.itos: Dict[int, str] = {idx: token for token, idx in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        unique_chars = sorted(set(text))
        tokens = ["<pad>", "<unk>"] + unique_chars
        return cls(tokens=tokens)

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> List[int]:
        unk_id = self.stoi["<unk>"]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            if token in {"<pad>", "<unk>"}:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tokens": self.tokens}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        in_path = Path(path)
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        return cls(tokens=payload["tokens"])
