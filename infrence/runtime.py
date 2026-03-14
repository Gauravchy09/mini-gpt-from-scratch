from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import json

import torch
from tokenizers import Tokenizer

from model.notebook_model import build_model_config
from models.gpt_model import MiniGPT
from utils.device import get_device


@dataclass
class RuntimeBundle:
    model: MiniGPT
    tokenizer: Tokenizer
    device: torch.device
    export_dir: Path


def remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Notebook export names differ from repo model module names.
    key_map = {
        "embedding.token_emb.weight": "embedding.token_embedding.weight",
        "embedding.pos_emb.weight": "embedding.position_embedding.weight",
    }
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        remapped[key_map.get(key, key)] = value
    return remapped


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def find_export_dirs(root_dir: str | Path, preferred_names: tuple[str, ...] = ("run_02", "run_01")) -> list[Path]:
    root = Path(root_dir)
    found: list[Path] = []

    default_export = root / "artifacts" / "mini_gpt_demo_export"
    if (default_export / "mini_gpt_state.pt").exists():
        found.append(default_export)

    for parent in sorted(root.glob("run_*"), reverse=True):
        for name in preferred_names:
            candidate = parent / name
            if (candidate / "mini_gpt_state.pt").exists():
                found.append(candidate)

        for child in sorted(parent.glob("run_*"), reverse=True):
            if (child / "mini_gpt_state.pt").exists():
                found.append(child)

    unique = _dedupe_paths(found)
    preferred = [p for p in unique if p.name in preferred_names]
    non_preferred = [p for p in unique if p.name not in preferred_names]

    preferred.sort(key=lambda p: preferred_names.index(p.name) if p.name in preferred_names else 999)
    return preferred + non_preferred


def pick_default_export(export_dirs: list[Path]) -> Path | None:
    if not export_dirs:
        return None
    for path in export_dirs:
        if path.name == "run_02":
            return path
    return export_dirs[0]


@lru_cache(maxsize=8)
def load_runtime_bundle(export_dir: str, device_name: str = "auto") -> RuntimeBundle:
    export_path = Path(export_dir)
    model_path = export_path / "mini_gpt_state.pt"
    config_path = export_path / "mini_gpt_config.json"
    tokenizer_path = export_path / "tokenizer" / "tokenizer.json"

    if not model_path.exists() or not config_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            "Invalid export folder. Expected mini_gpt_state.pt, mini_gpt_config.json, and tokenizer/tokenizer.json"
        )

    config = build_model_config(config_path)
    model = MiniGPT(config)
    tokenizer = Tokenizer.from_file(tokenizer_path.as_posix())

    device = get_device(device_name)
    loaded = torch.load(model_path, map_location=device)
    state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
    state_dict = remap_state_dict_keys(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return RuntimeBundle(model=model, tokenizer=tokenizer, device=device, export_dir=export_path)


def _apply_top_p(next_logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(next_logits, dtype=torch.bool)
    remove_mask.scatter_(1, sorted_indices, sorted_remove)
    return next_logits.masked_fill(remove_mask, float("-inf"))


def generate_from_export(
    export_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.92,
    repetition_penalty: float = 1.15,
    device_name: str = "auto",
) -> str:
    bundle = load_runtime_bundle(str(export_dir), device_name=device_name)

    encoded = bundle.tokenizer.encode(prompt)
    input_ids = encoded.ids
    if not input_ids:
        return ""

    x = torch.tensor([input_ids], dtype=torch.long, device=bundle.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = x[:, -bundle.model.config.context_length :]
            logits, _ = bundle.model(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if repetition_penalty > 1.0:
                for batch_idx in range(x.size(0)):
                    seen_tokens = torch.unique(x[batch_idx])
                    next_logits[batch_idx, seen_tokens] = next_logits[batch_idx, seen_tokens] / repetition_penalty

            if top_k > 0:
                values, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                cutoff = values[:, [-1]]
                next_logits = torch.where(
                    next_logits < cutoff,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            if 0.0 < top_p < 1.0:
                next_logits = _apply_top_p(next_logits, top_p)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    return bundle.tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
