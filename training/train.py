from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data.dataset import build_train_val_datasets, load_text
from models.gpt_model import GPTConfig, MiniGPT
from tokenizer.tokenizer import SimpleCharTokenizer
from training.trainer import Trainer, TrainerConfig
from utils.device import get_device
from utils.logger import setup_logger
from utils.seed import set_seed


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mini-GPT model")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--training_config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--data_path", type=str, default="data/raw/input.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--log_file", type=str, default="experiments/logs/train.log")
    parser.add_argument("--skip_demo_data", action="store_true")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--train_split", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--max_steps_per_epoch", type=int)

    parser.add_argument("--embedding_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--dropout", type=float)

    return parser.parse_args(argv)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def build_training_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "eval_interval": args.eval_interval,
        "log_interval": args.log_interval,
        "grad_clip": args.grad_clip,
        "train_split": args.train_split,
        "device": args.device,
        "max_steps_per_epoch": args.max_steps_per_epoch,
    }


def build_model_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "context_length": args.context_length,
        "dropout": args.dropout,
    }


def maybe_create_demo_data(data_path: Path) -> None:
    if data_path.exists():
        return

    data_path.parent.mkdir(parents=True, exist_ok=True)
    sample_text = (
        "deep learning is fun. "
        "transformers are powerful models for sequence tasks. "
        "this is a tiny demo dataset for mini gpt training. "
    )
    data_path.write_text(sample_text * 200, encoding="utf-8")


def main() -> None:
    args = parse_args()

    model_config_path = resolve_path(args.model_config)
    training_config_path = resolve_path(args.training_config)
    raw_data_path = resolve_path(args.data_path)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    log_file = resolve_path(args.log_file)

    if not args.skip_demo_data:
        # This keeps the repo runnable even before the user downloads a real dataset.
        maybe_create_demo_data(raw_data_path)

    model_cfg = load_yaml(model_config_path)
    train_cfg = load_yaml(training_config_path)
    model_cfg = apply_overrides(model_cfg, build_model_overrides(args))
    train_cfg = apply_overrides(train_cfg, build_training_overrides(args))

    set_seed(int(train_cfg.get("seed", 42)))
    logger = setup_logger(
        name="mini_gpt_train",
        log_file=log_file,
    )

    text = load_text(raw_data_path)

    tokenizer = SimpleCharTokenizer()
    tokenizer.fit(text)
    token_ids = tokenizer.encode(text)

    # The tokenizer decides the final vocabulary size used by the model.
    model_cfg["vocab_size"] = tokenizer.vocab_size
    gpt_config = GPTConfig(**model_cfg)

    train_dataset, val_dataset = build_train_val_datasets(
        token_ids=token_ids,
        block_size=gpt_config.context_length,
        train_split=float(train_cfg.get("train_split", 0.9)),
    )

    model = MiniGPT(gpt_config)

    trainer_config = TrainerConfig(
        epochs=int(train_cfg.get("epochs", 5)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        eval_interval=int(train_cfg.get("eval_interval", 200)),
        log_interval=int(train_cfg.get("log_interval", 50)),
        max_steps_per_epoch=train_cfg.get("max_steps_per_epoch"),
        checkpoint_dir=str(checkpoint_dir),
    )

    tokenizer.save(ROOT_DIR / "tokenizer" / "vocab.json")
    logger.info("Tokenizer vocabulary size: %d", tokenizer.vocab_size)
    logger.info(
        "Model config | embedding_dim=%d num_heads=%d num_layers=%d context_length=%d dropout=%.3f",
        gpt_config.embedding_dim,
        gpt_config.num_heads,
        gpt_config.num_layers,
        gpt_config.context_length,
        gpt_config.dropout,
    )
    logger.info(
        "Training config | epochs=%d batch_size=%d learning_rate=%.6f train_split=%.2f",
        trainer_config.epochs,
        trainer_config.batch_size,
        trainer_config.learning_rate,
        float(train_cfg.get("train_split", 0.9)),
    )

    device = get_device(train_cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        device=device,
        logger=logger,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
