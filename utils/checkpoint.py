from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, out_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    in_path = Path(path)
    payload = torch.load(in_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload
