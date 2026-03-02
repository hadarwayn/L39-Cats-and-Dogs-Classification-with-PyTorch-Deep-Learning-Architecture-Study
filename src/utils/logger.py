"""
Logging with a Ring Buffer — keeps log files from growing forever.

A ring buffer is like a circular notepad: when you fill the last page,
you start writing over the first page. This prevents log files from
eating up all your disk space.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class RingBufferHandler(RotatingFileHandler):
    """
    A rotating file handler that limits total log size.

    When the log file hits max_bytes, it rolls over to a backup file.
    Old backups are deleted, keeping only backup_count files.
    """

    def __init__(
        self,
        filename: str,
        max_bytes: int = 5_242_880,  # 5 MB
        backup_count: int = 2,
    ) -> None:
        """
        Set up the ring buffer handler.

        Args:
            filename: path to the log file
            max_bytes: maximum size before rotating (default 5 MB)
            backup_count: how many old log files to keep
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )


def setup_logger(
    name: str = "L39",
    config_path: str | None = None,
) -> logging.Logger:
    """
    Create and configure a logger with console + file output.

    Args:
        name: logger name (shows up in log lines)
        config_path: optional path to log_config.json

    Returns:
        A ready-to-use logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    # Load config or use defaults
    config = _load_config(config_path)

    logger.setLevel(logging.DEBUG)

    # Console handler — shows INFO and above
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, config.get("console_level", "INFO")))
    console.setFormatter(_make_formatter(config))
    logger.addHandler(console)

    # File handler — ring buffer, logs everything
    log_file = config.get("log_file", "logs/training.log")
    ring = config.get("ring_buffer", {})
    file_handler = RingBufferHandler(
        filename=log_file,
        max_bytes=ring.get("max_bytes", 5_242_880),
        backup_count=ring.get("backup_count", 2),
    )
    file_handler.setLevel(getattr(logging, config.get("file_level", "DEBUG")))
    file_handler.setFormatter(_make_formatter(config))
    logger.addHandler(file_handler)

    return logger


def _load_config(config_path: str | None) -> dict:
    """Load JSON config from file, or return empty dict."""
    if config_path is None:
        default = Path(__file__).resolve().parent.parent.parent / "logs" / "config" / "log_config.json"
        config_path = str(default)
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _make_formatter(config: dict) -> logging.Formatter:
    """Build a log formatter from config."""
    fmt_cfg = config.get("formatters", {}).get("standard", {})
    return logging.Formatter(
        fmt=fmt_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
        datefmt=fmt_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )


def print_log_status(logger: logging.Logger) -> None:
    """Print info about the logger's handlers — useful for debugging."""
    print(f"\nLogger '{logger.name}' has {len(logger.handlers)} handler(s):")
    for handler in logger.handlers:
        htype = type(handler).__name__
        level = logging.getLevelName(handler.level)
        print(f"  - {htype} (level={level})")
