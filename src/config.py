"""
Central configuration for L39 - Cats vs Dogs Classification.

All magic numbers, paths, and constants live here.
Think of this as the "control panel" for the entire project.
"""

from pathlib import Path

import matplotlib
import torch
import yaml


# --- Environment Detection ---
def is_colab() -> bool:
    """Check if we are running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

# Set non-interactive matplotlib backend for local/headless runs
if not is_colab():
    matplotlib.use("Agg")


def get_device() -> torch.device:
    """Pick the best available device — GPU if possible, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- Paths (all relative to project root) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
EXAMPLES_DIR = RESULTS_DIR / "examples"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# --- Load YAML settings ---
_settings_path = CONFIG_DIR / "settings.yaml"
if _settings_path.exists():
    with open(_settings_path, "r") as f:
        _settings = yaml.safe_load(f)
else:
    _settings = {}

# --- Image & Dataset ---
IMAGE_SIZE: int = 150
NUM_CLASSES: int = 2
CLASS_NAMES: list[str] = ["cat", "dog"]

# --- Training Hyperparameters ---
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
RANDOM_SEED: int = 42
EARLY_STOPPING_PATIENCE: int = 5
MAX_LOCAL_TRAINING_SECONDS: int = 7200  # 2-hour hard limit

# --- Environment-specific defaults ---
ENVIRONMENT: str = "colab" if is_colab() else "local"
DEVICE: torch.device = get_device()

DEFAULT_EPOCHS: int = 15 if ENVIRONMENT == "colab" else 10

# Dataset size limits — small for local CPU, full for Colab GPU
LOCAL_MAX_TRAIN: int = 2000
LOCAL_MAX_VAL: int = 1000
COLAB_MAX_TRAIN: int = 20000
COLAB_MAX_VAL: int = 5000
MAX_TRAIN_SAMPLES: int = COLAB_MAX_TRAIN if ENVIRONMENT == "colab" else LOCAL_MAX_TRAIN
MAX_VAL_SAMPLES: int = COLAB_MAX_VAL if ENVIRONMENT == "colab" else LOCAL_MAX_VAL

# --- Visualization ---
GRAPH_DPI: int = 300
FIGURE_SIZE: tuple[int, int] = (10, 6)

# Color scheme for consistent graphs (one color per model)
MODEL_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Research group colors
GROUP_COLORS: dict[str, str] = {
    "A": "#1f77b4",  # Blue — Depth study
    "B": "#2ca02c",  # Green — Regularization
    "C": "#ff7f0e",  # Orange — Architecture
    "D": "#d62728",  # Red — Advanced
}

# --- Model Registry Info ---
MODEL_NAMES: dict[int, str] = {
    1: "Baseline CNN",
    2: "Shallow CNN",
    3: "Deep CNN",
    4: "Wide CNN",
    5: "Small FC CNN",
    6: "Dropout CNN",
    7: "BatchNorm CNN",
    8: "LeNet-Style CNN",
    9: "Transfer ResNet18",
    10: "Lightweight CNN",
}

MODEL_GROUPS: dict[int, str] = {
    1: "A", 2: "A", 3: "A", 4: "A",
    5: "C", 6: "B", 7: "B", 8: "C",
    9: "D", 10: "D",
}
