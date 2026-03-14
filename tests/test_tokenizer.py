from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tokenizer.tokenizer import SimpleCharTokenizer


def test_tokenizer_encode_decode_roundtrip() -> None:
    text = "hello transformer"
    tokenizer = SimpleCharTokenizer()
    tokenizer.fit(text)

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text
    assert tokenizer.vocab_size >= len(set(text))
