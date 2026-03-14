from __future__ import annotations

import argparse
from pathlib import Path

from infrence.runtime import find_export_dirs, generate_from_export, pick_default_export

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from notebook-exported Mini-GPT artifacts")
    parser.add_argument("--prompt", type=str, default="When the astronaut landed on Mars, she discovered")
    parser.add_argument("--export_dir", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.export_dir:
        export_dir = Path(args.export_dir)
        if not export_dir.is_absolute():
            export_dir = ROOT_DIR / export_dir
    else:
        export_dir = pick_default_export(find_export_dirs(ROOT_DIR))
        if export_dir is None:
            raise FileNotFoundError("No export folder found. Expected run_02-*/run_02 with mini_gpt_state.pt")

    generated = generate_from_export(
        export_dir=export_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device_name=args.device,
    )

    print("Export folder:", export_dir)
    print("Prompt:", args.prompt)
    print("Generated Text:")
    print(generated)


if __name__ == "__main__":
    main()
