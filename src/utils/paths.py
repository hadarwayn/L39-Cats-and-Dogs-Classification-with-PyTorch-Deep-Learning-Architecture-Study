"""
Path utilities — every directory the project needs.

All paths are built from the project root using pathlib,
so they work on any operating system.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root folder."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Where downloaded images are stored."""
    path = get_project_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_dir() -> Path:
    """Top-level results folder."""
    path = get_project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_graphs_dir() -> Path:
    """Where all graph PNG files are saved."""
    path = get_results_dir() / "graphs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_tables_dir() -> Path:
    """Where CSV result tables are saved."""
    path = get_results_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    """Where trained model .pth files are saved."""
    path = get_results_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir() -> Path:
    """Where log files are written."""
    path = get_project_root() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
