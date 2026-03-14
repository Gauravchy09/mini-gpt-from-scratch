from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.gpt_model import GPTConfig, MiniGPT
from tokenizer.tokenizer import SimpleCharTokenizer
from utils.checkpoint import load_checkpoint
from utils.device import get_device


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with Mini-GPT")
    parser.add_argument("--prompt", type=str, default="deep learning is")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/final.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_cfg_dict = load_yaml(ROOT_DIR / "configs" / "model_config.yaml")
    tokenizer = SimpleCharTokenizer.load(ROOT_DIR / "tokenizer" / "vocab.json")
    model_cfg_dict["vocab_size"] = tokenizer.vocab_size

    config = GPTConfig(**model_cfg_dict)
    model = MiniGPT(config)

    device = get_device("auto")
    checkpoint_path = ROOT_DIR / args.checkpoint
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device)
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}. Using random weights.")

    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            input_ids=x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    generated_text = tokenizer.decode(out[0].tolist())
    print("Prompt:", args.prompt)
    print("Generated Text:")
    print(generated_text)


if __name__ == "__main__":
    main()
