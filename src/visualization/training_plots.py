"""
Training history plots — accuracy and loss curves per model.

These graphs show how the model improved over time (epochs).
If the training line is much higher than validation, the model
is "overfitting" — memorising answers instead of learning.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.config import GRAPH_DPI
from src.utils.paths import get_graphs_dir


def plot_training_summary(
    history: dict,
    model_name: str,
) -> None:
    """
    Create a 2-panel figure: accuracy + loss curves.

    Shows both training and validation on the same axes so
    you can spot overfitting at a glance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_acc"]) + 1)

    # --- Accuracy panel ---
    ax1.plot(epochs, history["train_acc"], "b-o", label="Train", markersize=4)
    ax1.plot(epochs, history["val_acc"], "r-o", label="Validation", markersize=4)
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_val = max(history["val_acc"])
    ax1.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"Best: {best_val:.1%} (ep {best_epoch})",
        xy=(best_epoch, best_val), fontsize=9,
        xytext=(5, -15), textcoords="offset points",
    )
    ax1.set_title(f"Accuracy — {model_name}", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Loss panel ---
    ax2.plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    ax2.plot(epochs, history["val_loss"], "r-o", label="Validation", markersize=4)
    ax2.set_title(f"Loss — {model_name}", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(get_graphs_dir() / f"training_{safe_name}.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_accuracy_curves(history: dict, model_name: str) -> None:
    """Plot train vs validation accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_acc"]) + 1)

    ax.plot(epochs, history["train_acc"], "b-o", label="Train", markersize=4)
    ax.plot(epochs, history["val_acc"], "r-o", label="Validation", markersize=4)

    ax.set_title(f"Accuracy — {model_name}", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(get_graphs_dir() / f"accuracy_{safe_name}.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_loss_curves(history: dict, model_name: str) -> None:
    """Plot train vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    ax.plot(epochs, history["val_loss"], "r-o", label="Validation", markersize=4)

    ax.set_title(f"Loss — {model_name}", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(get_graphs_dir() / f"loss_{safe_name}.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()
