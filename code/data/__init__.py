"""Data management, quality control, preprocessing, and statistics package."""

from .loader import get_mat_files, record_stream_generator, load_single_mat_part
from .quality_control import (
    validate_ppg_signal,
    validate_abp_signal,
    perform_record_quality_control,
)
from .preprocessing import (
    apply_offline_research_ppg_filter,
    extract_physiological_bp_targets,
)
from .dataset_statistics import compute_dataset_qc_statistics
from .windowing import slice_record_into_windows, calculate_expected_window_count
from .window_quality import validate_ppg_window
from .target_generation import (
    normalize_raw_filtered,
    normalize_zscore,
    normalize_robust,
    extract_window_abp_targets,
    evaluate_window_quality_status,
)

__all__ = [
    "get_mat_files",
    "record_stream_generator",
    "load_single_mat_part",
    "validate_ppg_signal",
    "validate_abp_signal",
    "perform_record_quality_control",
    "apply_offline_research_ppg_filter",
    "extract_physiological_bp_targets",
    "compute_dataset_qc_statistics",
    "slice_record_into_windows",
    "calculate_expected_window_count",
    "validate_ppg_window",
    "normalize_raw_filtered",
    "normalize_zscore",
    "normalize_robust",
    "extract_window_abp_targets",
    "evaluate_window_quality_status",
]

