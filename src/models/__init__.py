"""
Model Registry — one place to access all 10 CNN architectures.

Usage:
    from src.models import get_model, get_all_model_ids
    model = get_model(1)  # returns BaselineCNN instance
"""

from src.models.model_01_baseline import BaselineCNN
from src.models.model_02_shallow import ShallowCNN
from src.models.model_03_deep import DeepCNN
from src.models.model_04_wide import WideCNN
from src.models.model_05_small_fc import SmallFCCNN
from src.models.model_06_dropout import DropoutCNN
from src.models.model_07_batchnorm import BatchNormCNN
from src.models.model_08_lenet import LeNetCNN
from src.models.model_09_transfer import TransferResNet18
from src.models.model_10_lightweight import LightweightCNN

# Maps model_id -> (class, display_name, short_description, group)
MODEL_REGISTRY: dict[int, tuple] = {
    1:  (BaselineCNN,       "Baseline CNN",      "Keras conversion, 4 conv layers",     "A"),
    2:  (ShallowCNN,        "Shallow CNN",       "Minimal 2 conv layers",               "A"),
    3:  (DeepCNN,           "Deep CNN",          "6 conv layers, tests depth",           "A"),
    4:  (WideCNN,           "Wide CNN",          "3 wide layers (64-128-256)",           "A"),
    5:  (SmallFCCNN,        "Small FC CNN",      "Compact 128-neuron classifier",        "C"),
    6:  (DropoutCNN,        "Dropout CNN",       "Baseline + Dropout regularisation",    "B"),
    7:  (BatchNormCNN,      "BatchNorm CNN",     "Baseline + BatchNorm normalisation",   "B"),
    8:  (LeNetCNN,          "LeNet-Style CNN",   "Classic 5x5 kernels, AvgPool",         "C"),
    9:  (TransferResNet18,  "Transfer ResNet18", "Pretrained ResNet18, frozen backbone", "D"),
    10: (LightweightCNN,    "Lightweight CNN",   "Global AvgPool, ~30K params",          "D"),
}


def get_model(model_id: int, input_channels: int = 3) -> object:
    """
    Create and return a model instance by its ID (1-10).

    Args:
        model_id: which architecture to build (1 through 10)
        input_channels: 3 for RGB, 1 for grayscale

    Returns:
        An instantiated PyTorch model ready for training.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id={model_id}. Valid: 1-10")

    model_class = MODEL_REGISTRY[model_id][0]

    # Model 9 (Transfer) only works with RGB
    if model_id == 9 and input_channels != 3:
        raise ValueError("TransferResNet18 requires RGB (input_channels=3)")

    return model_class(input_channels=input_channels)


def get_model_info(model_id: int) -> dict[str, str]:
    """Return name, description, and group for a model."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id={model_id}. Valid: 1-10")
    _, name, desc, group = MODEL_REGISTRY[model_id]
    return {"name": name, "description": desc, "group": group}


def get_all_model_ids() -> list[int]:
    """Return a sorted list of all available model IDs."""
    return sorted(MODEL_REGISTRY.keys())
