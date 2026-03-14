from __future__ import annotations

import torch


def get_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()

    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto mode picks CUDA when available, otherwise CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
