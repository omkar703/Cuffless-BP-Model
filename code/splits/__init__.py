"""Splits package for leakage-safe record-level dataset partitioning."""

from .record_split import (
    create_record_level_split,
    run_leakage_and_integrity_assertions,
    compute_split_distribution_summary,
)

__all__ = [
    "create_record_level_split",
    "run_leakage_and_integrity_assertions",
    "compute_split_distribution_summary",
]
