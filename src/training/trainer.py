"""
Training loop with time limit, early stopping, and progress bars.

The Trainer handles the full process of teaching a model:
1. Show images to the model (forward pass)
2. Check how wrong the model was (loss)
3. Adjust the model's weights to be less wrong (backward pass)
4. Repeat for many epochs (full passes through the data)
"""

import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEFAULT_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    ENVIRONMENT,
    LEARNING_RATE,
    MAX_LOCAL_TRAINING_SECONDS,
)
from src.utils.helpers import format_time
from src.utils.logger import setup_logger

logger = setup_logger("trainer")


class Trainer:
    """Trains a model and tracks metrics per epoch."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = LEARNING_RATE,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5,
        )
        self.best_val_acc = 0.0

    def fit(self, num_epochs: int = DEFAULT_EPOCHS) -> dict:
        """
        Train the model for num_epochs and return history.

        Returns a dict with lists: train_acc, val_acc, train_loss, val_loss.
        Stops early if val_loss doesn't improve, or if time limit hit.
        """
        history: dict[str, list] = {
            "train_acc": [], "val_acc": [],
            "train_loss": [], "val_loss": [],
        }
        patience_counter = 0
        best_val_loss = float("inf")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # Check time limit (local environment only)
            elapsed = time.time() - start_time
            if ENVIRONMENT == "local" and elapsed > MAX_LOCAL_TRAINING_SECONDS:
                logger.warning("2-hour time limit reached. Stopping training.")
                break

            # Train one epoch
            train_loss, train_acc = self._train_one_epoch(epoch, num_epochs)
            val_loss, val_acc = self._validate()

            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Learning rate scheduler
            self.scheduler.step(val_loss)

            # Track best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d/%d — Train: %.1f%% | Val: %.1f%% | "
                "Loss: %.4f | Time: %s",
                epoch, num_epochs, train_acc * 100, val_acc * 100,
                val_loss, format_time(elapsed),
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered (patience=%d).",
                            EARLY_STOPPING_PATIENCE)
                break

        total_time = time.time() - start_time
        history["total_time"] = total_time
        history["epochs_completed"] = len(history["train_acc"])
        logger.info("Training complete in %s", format_time(total_time))
        return history

    def _train_one_epoch(self, epoch: int, total: int) -> tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total} [Train]",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / total_samples
        accuracy = correct / total_samples
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Run validation pass. Returns (avg_loss, accuracy)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        avg_loss = running_loss / total_samples
        accuracy = correct / total_samples
        return avg_loss, accuracy
