"""
Main entry point — trains and evaluates CNN models locally.

Usage:
    python main.py                  # train all 10 models (RGB, default epochs)
    python main.py --model 1        # train only model 1
    python main.py --mode grayscale # use grayscale images
    python main.py --epochs 5       # override epoch count
"""

import argparse

import torch

from src.config import DEFAULT_EPOCHS, DEVICE, ENVIRONMENT, RANDOM_SEED
from src.data.loader import create_dataloaders
from src.models import get_all_model_ids, get_model, get_model_info
from src.training.evaluator import Evaluator
from src.training.results import print_final_summary, save_results_summary
from src.training.trainer import Trainer
from src.utils.helpers import count_parameters, print_model_summary, set_all_seeds
from src.utils.logger import setup_logger
from src.visualization.comparison_plots import (
    plot_accuracy_comparison, plot_accuracy_vs_params,
    plot_group_comparison, plot_loss_comparison,
    plot_params_comparison, plot_time_comparison,
)
from src.visualization.confusion_plots import plot_confusion_matrix
from src.visualization.sample_plots import (
    plot_class_distribution, plot_predictions_grid, plot_sample_images,
)
from src.visualization.training_plots import plot_training_summary

logger = setup_logger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="L39 Cats vs Dogs CNN Training")
    parser.add_argument("--model", type=str, default="all",
                        help="Model ID (1-10) or 'all'")
    parser.add_argument("--mode", type=str, default="rgb",
                        choices=["rgb", "grayscale"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args()


def main() -> None:
    """Orchestrate the full training and evaluation pipeline."""
    args = parse_args()
    set_all_seeds(RANDOM_SEED)

    logger.info("Environment: %s | Device: %s | Mode: %s | Epochs: %d",
                ENVIRONMENT, DEVICE, args.mode, args.epochs)

    # Determine which models to run
    if args.model == "all":
        model_ids = get_all_model_ids()
    else:
        model_ids = [int(args.model)]

    if args.mode == "grayscale":
        model_ids = [m for m in model_ids if m != 9]
        logger.info("Skipping model 9 (Transfer) — requires RGB input")

    # Create dataloaders (shared across all models)
    train_loader, val_loader = create_dataloaders(mode=args.mode)
    plot_sample_images(train_loader)
    if hasattr(train_loader.dataset, "samples"):
        plot_class_distribution(train_loader.dataset)

    # Train and evaluate each model
    all_results = _train_all_models(model_ids, args, train_loader, val_loader)

    # Cross-model comparison (only if multiple models ran)
    if len(all_results) > 1:
        logger.info("\n%s Generating Comparison Charts %s", "=" * 15, "=" * 15)
        for plot_fn in [plot_accuracy_comparison, plot_loss_comparison,
                        plot_params_comparison, plot_time_comparison,
                        plot_accuracy_vs_params, plot_group_comparison]:
            plot_fn(all_results)

    save_results_summary(all_results)
    print_final_summary(all_results)
    logger.info("All done! Results saved to results/")


def _train_all_models(
    model_ids: list[int], args: argparse.Namespace,
    train_loader: object, val_loader: object,
) -> dict:
    """Train and evaluate each model, returning collected results."""
    all_results: dict = {}
    channels = 3 if args.mode == "rgb" else 1

    for model_id in model_ids:
        info = get_model_info(model_id)
        model_name = info["name"]
        logger.info("\n%s Training Model %d: %s %s",
                    "=" * 20, model_id, model_name, "=" * 20)
        try:
            result = _train_single_model(
                model_id, model_name, info, channels,
                args.epochs, train_loader, val_loader,
            )
            all_results[model_name] = result
        except Exception as e:
            logger.error("Model %d (%s) failed: %s", model_id, model_name, e)
    return all_results


def _train_single_model(
    model_id: int, model_name: str, info: dict,
    channels: int, epochs: int,
    train_loader: object, val_loader: object,
) -> dict:
    """Train one model, evaluate it, generate plots, return results."""
    model = get_model(model_id, input_channels=channels).to(DEVICE)
    print_model_summary(model, model_name)
    params = count_parameters(model)

    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    history = trainer.fit(num_epochs=epochs)
    metrics = Evaluator.evaluate_model(model, val_loader, DEVICE)

    # Per-model visualizations
    plot_training_summary(history, model_name)
    plot_confusion_matrix(metrics["confusion_matrix"], model_name)
    wrong = Evaluator.get_misclassified_samples(model, val_loader, DEVICE, num=12)
    if wrong[0]:
        plot_predictions_grid(torch.stack(wrong[0]), wrong[1], wrong[2],
                              model_name=model_name)

    logger.info("Model %d done — Val Accuracy: %.1f%%",
                model_id, metrics["accuracy"] * 100)

    return {
        "model_id": model_id, "accuracy": metrics["accuracy"],
        "train_acc": history["train_acc"], "val_acc": history["val_acc"],
        "train_loss": history["train_loss"], "val_loss": history["val_loss"],
        "total_params": params["total"], "trainable_params": params["trainable"],
        "total_time": history["total_time"],
        "epochs_completed": history["epochs_completed"],
        "group": info["group"], "confusion_matrix": metrics["confusion_matrix"],
    }


if __name__ == "__main__":
    main()
