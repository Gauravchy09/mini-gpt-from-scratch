from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from training.train import build_model_overrides, build_training_overrides, parse_args


def test_train_cli_parses_demo_overrides() -> None:
    args = parse_args(
        [
            "--epochs",
            "1",
            "--batch_size",
            "8",
            "--context_length",
            "32",
            "--max_steps_per_epoch",
            "10",
            "--skip_demo_data",
        ]
    )

    training_overrides = build_training_overrides(args)
    model_overrides = build_model_overrides(args)

    assert args.skip_demo_data is True
    assert training_overrides["epochs"] == 1
    assert training_overrides["batch_size"] == 8
    assert training_overrides["max_steps_per_epoch"] == 10
    assert model_overrides["context_length"] == 32
