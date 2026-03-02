"""
Sample image visualizations — see what the data looks like.

These plots help verify that images loaded correctly and let
us visually inspect model predictions later.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import CLASS_NAMES, GRAPH_DPI
from src.utils.paths import get_graphs_dir


def plot_sample_images(
    dataloader: torch.utils.data.DataLoader,
    num_images: int = 16,
) -> None:
    """
    Show a grid of sample images from the dataloader.

    Displays the first num_images images with their labels
    so you can quickly check the data looks right.
    """
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(num_images):
        img = _tensor_to_image(images[i])
        axes[i].imshow(img)
        axes[i].set_title(CLASS_NAMES[labels[i].item()], fontsize=12)
        axes[i].axis("off")

    # Hide unused axes
    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Images from Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(get_graphs_dir() / "sample_images.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_class_distribution(dataset: torch.utils.data.Dataset) -> None:
    """
    Bar chart showing how many cats and dogs are in the dataset.

    A balanced dataset (roughly 50/50) trains better models.
    """
    labels = [label for _, label in dataset.samples]
    counts = [labels.count(0), labels.count(1)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=["#ff7f0e", "#1f77b4"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(count), ha="center", fontweight="bold")

    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(get_graphs_dir() / "class_distribution.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_predictions_grid(
    images: torch.Tensor,
    true_labels: list[int],
    pred_labels: list[int],
    num_images: int = 12,
    model_name: str = "",
) -> None:
    """
    Grid of images showing what the model predicted vs the truth.

    Green title = correct prediction, Red title = wrong prediction.
    """
    num_images = min(num_images, len(images))
    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(num_images):
        img = _tensor_to_image(images[i])
        axes[i].imshow(img)
        true_name = CLASS_NAMES[true_labels[i]]
        pred_name = CLASS_NAMES[pred_labels[i]]
        correct = true_labels[i] == pred_labels[i]
        color = "green" if correct else "red"
        axes[i].set_title(f"True: {true_name}\nPred: {pred_name}",
                          color=color, fontsize=10)
        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    title = f"Predictions — {model_name}" if model_name else "Predictions"
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower() or "model"
    plt.savefig(get_graphs_dir() / f"predictions_{safe_name}.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to displayable NumPy array."""
    img = tensor.cpu().numpy()
    # Un-normalise from [-1, 1] to [0, 1]
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    if img.shape[0] == 1:
        # Grayscale — squeeze channel dim
        return img.squeeze(0)
    # RGB — move channel to last axis
    return np.transpose(img, (1, 2, 0))
