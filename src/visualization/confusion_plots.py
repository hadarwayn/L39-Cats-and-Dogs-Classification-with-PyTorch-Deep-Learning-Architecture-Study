"""
Confusion matrix plots — see where the model gets confused.

A confusion matrix is a 2x2 table:
                   Predicted Cat    Predicted Dog
  Actually Cat     [Correct!]       [Oops, wrong]
  Actually Dog     [Oops, wrong]    [Correct!]

The diagonal = correct predictions. Off-diagonal = mistakes.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import CLASS_NAMES, GRAPH_DPI
from src.utils.paths import get_graphs_dir


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
) -> None:
    """
    Draw a heatmap confusion matrix with counts and percentages.

    Args:
        cm: 2x2 numpy array from sklearn confusion_matrix
        model_name: used in the title and filename
    """
    total = cm.sum()
    # Build labels showing both count and percentage
    labels = np.array([
        [f"{val}\n({val / total:.1%})" for val in row]
        for row in cm
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=labels, fmt="",
        cmap="Blues", cbar=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(get_graphs_dir() / f"confusion_{safe_name}.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()
