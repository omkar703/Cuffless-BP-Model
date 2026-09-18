"""
Phase 4A: Shared utilities for seeding, logging, JSON I/O, and GPU memory reporting.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across random, numpy, and torch.
    Also enables CUDA deterministic mode if GPU is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger with console + optional file handlers.

    Args:
        name:         Logger name.
        log_dir:      Directory to write log file (optional).
        log_filename: Log file name (optional).
        level:        Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt_console = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fmt_file = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    if log_dir is not None and log_filename is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_filename, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt_file)
        logger.addHandler(fh)

    return logger


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Serialize a dict to a JSON file with pretty formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and return as dict."""
    with open(path, "r") as f:
        return json.load(f)


def gpu_memory_summary() -> str:
    """
    Return a formatted string of current GPU memory usage.
    Returns 'CUDA not available' if no GPU is present.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA not available."
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        return (
            f"GPU Memory: allocated={allocated:.1f} MB, "
            f"reserved={reserved:.1f} MB, "
            f"total={total:.0f} MB"
        )
    except ImportError:
        return "PyTorch not installed."


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Return a pretty-formatted string of metric values."""
    lines = [f"  {prefix}{k}: {v:.4f}" for k, v in metrics.items()]
    return "\n".join(lines)
