from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from training.loss import language_model_loss
from utils.checkpoint import save_checkpoint


@dataclass
class TrainerConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    grad_clip: float
    eval_interval: int
    log_interval: int
    max_steps_per_epoch: Optional[int] = None
    checkpoint_dir: str = "experiments/checkpoints"


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainerConfig,
        device: torch.device,
        logger,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.logger = logger

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=True,
        )

        self.global_step = 0
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _run_eval(self, max_batches: int = 20) -> float:
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                x = x.to(self.device)
                y = y.to(self.device)

                logits, _ = self.model(x)
                loss = language_model_loss(logits, y)
                losses.append(loss.item())

        self.model.train()
        if not losses:
            return float("nan")
        return sum(losses) / len(losses)

    def fit(self) -> None:
        self.model.to(self.device)
        self.model.train()

        best_val_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            for batch_idx, (x, y) in enumerate(self.train_loader, start=1):
                if self.config.max_steps_per_epoch is not None and batch_idx > self.config.max_steps_per_epoch:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits, _ = self.model(x)
                loss = language_model_loss(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    self.logger.info(
                        "epoch=%d step=%d train_loss=%.4f",
                        epoch,
                        self.global_step,
                        loss.item(),
                    )

                if self.global_step % self.config.eval_interval == 0:
                    val_loss = self._run_eval()
                    self.logger.info(
                        "epoch=%d step=%d val_loss=%.4f",
                        epoch,
                        self.global_step,
                        val_loss,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            path=Path(self.config.checkpoint_dir) / "best.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            step=self.global_step,
                            extra={"val_loss": val_loss, "epoch": epoch},
                        )

            epoch_val_loss = self._run_eval()
            self.logger.info(
                "epoch=%d completed | val_loss=%.4f",
                epoch,
                epoch_val_loss,
            )

        save_checkpoint(
            path=Path(self.config.checkpoint_dir) / "final.pt",
            model=self.model,
            optimizer=self.optimizer,
            step=self.global_step,
            extra={"best_val_loss": best_val_loss},
        )
        self.logger.info("Training completed.")
