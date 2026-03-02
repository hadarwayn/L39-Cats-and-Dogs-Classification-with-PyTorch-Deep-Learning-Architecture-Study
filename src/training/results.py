"""
Results saving and summary printing.

Handles writing results to JSON and printing the final summary table.
"""

import json

from src.utils.helpers import format_time
from src.utils.logger import setup_logger
from src.utils.paths import get_tables_dir

logger = setup_logger("results")


def save_results_summary(all_results: dict) -> None:
    """Save a JSON summary of all model results."""
    summary = {}
    for name, r in all_results.items():
        summary[name] = {
            "model_id": r["model_id"],
            "accuracy": round(r["accuracy"], 4),
            "total_params": r["total_params"],
            "trainable_params": r["trainable_params"],
            "total_time": round(r["total_time"], 1),
            "epochs_completed": r["epochs_completed"],
            "group": r["group"],
        }
    path = get_tables_dir() / "results_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results summary saved to %s", path)


def print_final_summary(all_results: dict) -> None:
    """Print a formatted table of all results, sorted by accuracy."""
    print(f"\n{'='*80}")
    print(f"{'FINAL RESULTS SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"{'Model':<22} {'Accuracy':>10} {'Params':>12} {'Time':>10} {'Epochs':>8}")
    print(f"{'-'*80}")

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    for name, r in sorted_results:
        params_str = f"{r['total_params']:,}"
        print(f"{name:<22} {r['accuracy']:>9.1%} {params_str:>12} "
              f"{format_time(r['total_time']):>10} {r['epochs_completed']:>8}")

    print(f"{'='*80}\n")
