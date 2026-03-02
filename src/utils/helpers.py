"""
Helper utilities used across the whole project.

Small, reusable functions that don't belong to any single module.
"""

import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Lock every random number generator to the same seed.

    This makes experiments reproducible — running the same code
    twice gives the same results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    Turn a number of seconds into a human-readable string.

    Examples: 65.3 -> "1m 5s", 3661 -> "1h 1m 1s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """
    Count how many numbers (parameters) a model has.

    Returns a dict with 'total' and 'trainable' counts.
    Think of parameters like tiny knobs the model adjusts while learning.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def print_model_summary(model: torch.nn.Module, model_name: str = "") -> None:
    """Print a short summary showing the model's layers and parameter count."""
    params = count_parameters(model)
    header = f" Model: {model_name} " if model_name else " Model Summary "
    print(f"\n{'='*50}")
    print(f"{header:=^50}")
    print(f"{'='*50}")
    print(model)
    print(f"\nTotal parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"{'='*50}\n")
