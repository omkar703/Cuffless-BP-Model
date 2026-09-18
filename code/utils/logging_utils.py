"""Logging utilities for the PPG Blood Pressure Capstone pipeline."""

import logging
import sys
from pathlib import Path
from config.config import LOGS_DIR

def setup_logger(name: str = "ppg_bp", log_filename: str = "dataset_analysis.log", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger that outputs to both stdout and a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if re-initialized
    if logger.handlers:
        return logger

    # Console formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File formatter
    log_path = LOGS_DIR / log_filename
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_format = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger
