"""
Cross-model comparison charts — compare all 10 architectures.

These are the "big picture" visualizations that answer:
- Which model is most accurate?
- Which trains fastest?
- Does bigger always mean better?
"""

import matplotlib.pyplot as plt
import numpy as np

from src.config import GRAPH_DPI, GROUP_COLORS, MODEL_COLORS
from src.utils.paths import get_graphs_dir


def plot_accuracy_comparison(all_results: dict) -> None:
    """Bar chart comparing final validation accuracy of all models."""
    _bar_chart(
        all_results,
        metric_key="accuracy",
        title="Validation Accuracy Comparison",
        ylabel="Accuracy",
        filename="comparison_accuracy.png",
        fmt=".1%",
        multiply=100,
    )


def plot_loss_comparison(all_results: dict) -> None:
    """Bar chart comparing final validation loss of all models."""
    names = list(all_results.keys())
    values = [r["val_loss"][-1] for r in all_results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values, color=MODEL_COLORS[:len(names)])
    _annotate_bars(ax, bars, fmt=".4f")
    ax.set_title("Final Validation Loss Comparison", fontweight="bold")
    ax.set_ylabel("Loss")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(get_graphs_dir() / "comparison_loss.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_params_comparison(all_results: dict) -> None:
    """Bar chart comparing parameter counts across models."""
    names = list(all_results.keys())
    values = [r["total_params"] for r in all_results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values, color=MODEL_COLORS[:len(names)])
    for bar, val in zip(bars, values):
        label = f"{val / 1e6:.2f}M" if val >= 1e6 else f"{val / 1e3:.1f}K"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=8)
    ax.set_title("Parameter Count Comparison", fontweight="bold")
    ax.set_ylabel("Parameters")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(get_graphs_dir() / "comparison_params.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_time_comparison(all_results: dict) -> None:
    """Bar chart comparing training times across models."""
    names = list(all_results.keys())
    values = [r["total_time"] for r in all_results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values, color=MODEL_COLORS[:len(names)])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_title("Training Time Comparison", fontweight="bold")
    ax.set_ylabel("Time (seconds)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(get_graphs_dir() / "comparison_time.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_accuracy_vs_params(all_results: dict) -> None:
    """Scatter plot: accuracy vs parameter count (bigger != better?)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, r) in enumerate(all_results.items()):
        group = r.get("group", "A")
        color = GROUP_COLORS.get(group, "#999999")
        ax.scatter(
            r["total_params"], r["accuracy"] * 100,
            color=color, s=100, zorder=5,
        )
        ax.annotate(name, (r["total_params"], r["accuracy"] * 100),
                     fontsize=7, ha="left", va="bottom")

    ax.set_title("Accuracy vs Parameter Count", fontweight="bold")
    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(True, alpha=0.3)

    # Legend for groups
    for group, color in GROUP_COLORS.items():
        ax.scatter([], [], color=color, label=f"Group {group}", s=80)
    ax.legend()

    plt.tight_layout()
    plt.savefig(get_graphs_dir() / "comparison_acc_vs_params.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def plot_group_comparison(all_results: dict) -> None:
    """Grouped bar chart: average accuracy per research group."""
    groups: dict[str, list[float]] = {}
    for r in all_results.values():
        g = r.get("group", "A")
        groups.setdefault(g, []).append(r["accuracy"] * 100)

    names = sorted(groups.keys())
    avgs = [np.mean(groups[g]) for g in names]
    colors = [GROUP_COLORS.get(g, "#999") for g in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"Group {g}" for g in names], avgs, color=colors)
    _annotate_bars(ax, bars, fmt=".1f", suffix="%")
    ax.set_title("Average Accuracy by Research Group", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(get_graphs_dir() / "comparison_groups.png", dpi=GRAPH_DPI)
    plt.show()
    plt.close()


# --- Private helpers ---

def _bar_chart(
    all_results: dict, metric_key: str, title: str,
    ylabel: str, filename: str, fmt: str = ".1f",
    multiply: float = 1,
) -> None:
    """Generic sorted bar chart helper."""
    names = list(all_results.keys())
    values = [all_results[n][metric_key] * multiply for n in names]

    # Sort by value descending
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, values, color=[MODEL_COLORS[i % 10] for i in order])
    _annotate_bars(ax, bars, fmt=fmt, suffix="%" if multiply == 100 else "")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(get_graphs_dir() / filename, dpi=GRAPH_DPI)
    plt.show()
    plt.close()


def _annotate_bars(
    ax: plt.Axes, bars: object,
    fmt: str = ".1f", suffix: str = "",
) -> None:
    """Add value labels on top of each bar."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f"{height:{fmt}}{suffix}",
            ha="center", va="bottom", fontsize=8,
        )
