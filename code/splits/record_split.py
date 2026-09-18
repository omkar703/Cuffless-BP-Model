"""
Leakage-Safe Record-Level Dataset Partitioning & Integrity Assertions.

Ensures zero patient/record leakage between Train, Validation, and Test sets.

CRITICAL SCIENTIFIC PRINCIPLES:
1. Partitioning is performed exclusively on UNIQUE `record_id`, NEVER on `window_id`.
2. Windows originating from the same record are guaranteed to stay within the same partition.
3. Record overlap between any pair of partitions is strictly zero.
4. "Record-level separation is used as the strongest available leakage-control
   mechanism; explicit subject identity is not available in the dataset."
"""

from typing import Dict, Any, Tuple, List, Set
from pathlib import Path
import numpy as np
import pandas as pd

from config.config import (
    SPLIT_RATIOS,
    SPLIT_RANDOM_SEED,
    RECORD_SPLIT_FILENAME,
    SPLITS_DIR,
    LEAKAGE_CONTROL_NOTE,
)
from utils.logging_utils import setup_logger

logger = setup_logger("record_split")


def create_record_level_split(
    records: List[str] | pd.Series | np.ndarray,
    train_ratio: float = SPLIT_RATIOS["train"],
    val_ratio: float = SPLIT_RATIOS["val"],
    test_ratio: float = SPLIT_RATIOS["test"],
    seed: int = SPLIT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Partitions a collection of unique record identifiers into Train, Validation, and Test.

    Parameters:
        records: Collection of unique record_id strings (e.g. 'part_01_record_000001')
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        seed: Random seed for deterministic reproducibility (default: 42)

    Returns:
        pd.DataFrame with columns ['record_id', 'part_id', 'split']
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    unique_records = np.array(sorted(list(set(records))))
    n_total = len(unique_records)

    # Deterministic permutation using fixed seed
    rng = np.random.RandomState(seed)
    shuffled_records = rng.permutation(unique_records)

    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    train_records = set(shuffled_records[:n_train])
    val_records = set(shuffled_records[n_train : n_train + n_val])
    test_records = set(shuffled_records[n_train + n_val :])

    # Basic partition assertion check
    assert len(train_records) + len(val_records) + len(test_records) == n_total
    assert len(train_records.intersection(val_records)) == 0
    assert len(train_records.intersection(test_records)) == 0
    assert len(val_records.intersection(test_records)) == 0

    rows = []
    for rec_id in unique_records:
        if rec_id in train_records:
            split_label = "train"
        elif rec_id in val_records:
            split_label = "val"
        else:
            split_label = "test"

        part_id = rec_id.split("_record_")[0] if "_record_" in rec_id else "unknown"
        rows.append({
            "record_id": rec_id,
            "part_id": part_id,
            "split": split_label,
        })

    split_df = pd.DataFrame(rows)
    logger.info(
        f"Record-level split created: Train={len(train_records)} ({len(train_records)/n_total:.1%}), "
        f"Val={len(val_records)} ({len(val_records)/n_total:.1%}), "
        f"Test={len(test_records)} ({len(test_records)/n_total:.1%})"
    )
    return split_df


def run_leakage_and_integrity_assertions(
    window_manifest_df: pd.DataFrame,
    record_split_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Executes all 9 mandatory automated integrity and leakage assertions.
    If ANY assertion fails, an AssertionError is raised immediately to halt the pipeline.

    Check 1: No window belongs to more than one record.
    Check 2: No record_id appears in multiple partitions.
    Check 3: No train record appears in validation (overlap == 0).
    Check 4: No train record appears in test (overlap == 0).
    Check 5: No validation record appears in test (overlap == 0).
    Check 6: Every window has exactly one parent record.
    Check 7: Every modeling-eligible window has valid PPG and valid ABP target.
    Check 8: No NaN/Inf exists in target columns (sbp, dbp, map) for modeling-eligible windows.
    Check 9: No ECG-derived column exists in the Phase 2 modeling manifest.
    """
    logger.info("Executing 9 critical leakage and integrity assertions...")

    # Build record sets from split table
    train_records: Set[str] = set(record_split_df[record_split_df["split"] == "train"]["record_id"])
    val_records: Set[str] = set(record_split_df[record_split_df["split"] == "val"]["record_id"])
    test_records: Set[str] = set(record_split_df[record_split_df["split"] == "test"]["record_id"])

    # Check 1: No window belongs to more than one record
    window_to_records = window_manifest_df.groupby("window_id")["record_id"].nunique()
    if (window_to_records > 1).any():
        bad_windows = window_to_records[window_to_records > 1].index.tolist()
        raise AssertionError(f"CHECK 1 FAILED: Windows associated with multiple records: {bad_windows[:5]}")
    logger.info("  ✓ Check 1 Passed: No window belongs to more than one record.")

    # Check 2: No record_id appears in multiple partitions
    rec_to_splits = record_split_df.groupby("record_id")["split"].nunique()
    if (rec_to_splits > 1).any():
        bad_recs = rec_to_splits[rec_to_splits > 1].index.tolist()
        raise AssertionError(f"CHECK 2 FAILED: Records assigned to multiple splits: {bad_recs[:5]}")
    logger.info("  ✓ Check 2 Passed: No record_id appears in multiple partitions.")

    # Check 3: No train record appears in validation
    train_val_overlap = train_records.intersection(val_records)
    if len(train_val_overlap) > 0:
        raise AssertionError(f"CHECK 3 FAILED: Train/Val record overlap: {len(train_val_overlap)} records!")
    logger.info("  ✓ Check 3 Passed: Train and Validation record overlap is exactly 0.")

    # Check 4: No train record appears in test
    train_test_overlap = train_records.intersection(test_records)
    if len(train_test_overlap) > 0:
        raise AssertionError(f"CHECK 4 FAILED: Train/Test record overlap: {len(train_test_overlap)} records!")
    logger.info("  ✓ Check 4 Passed: Train and Test record overlap is exactly 0.")

    # Check 5: No validation record appears in test
    val_test_overlap = val_records.intersection(test_records)
    if len(val_test_overlap) > 0:
        raise AssertionError(f"CHECK 5 FAILED: Val/Test record overlap: {len(val_test_overlap)} records!")
    logger.info("  ✓ Check 5 Passed: Validation and Test record overlap is exactly 0.")

    # Check 6: Every window has exactly one parent record
    null_parents = window_manifest_df["record_id"].isna().sum()
    if null_parents > 0:
        raise AssertionError(f"CHECK 6 FAILED: {null_parents} windows have missing parent record_id!")
    logger.info("  ✓ Check 6 Passed: Every window has exactly one parent record.")

    # Check 7: Every modeling-eligible window has valid PPG and valid ABP target
    eligible_df = window_manifest_df[window_manifest_df["modeling_eligible"] == True]
    if len(eligible_df) > 0:
        invalid_ppg_in_eligible = (~eligible_df["ppg_valid"]).sum()
        invalid_abp_in_eligible = (~eligible_df["abp_valid"]).sum()
        if invalid_ppg_in_eligible > 0 or invalid_abp_in_eligible > 0:
            raise AssertionError(
                f"CHECK 7 FAILED: Modeling-eligible subset contains invalid data: "
                f"invalid_ppg={invalid_ppg_in_eligible}, invalid_abp={invalid_abp_in_eligible}"
            )
    logger.info("  ✓ Check 7 Passed: All modeling-eligible windows have valid PPG and valid ABP.")

    # Check 8: No NaN/Inf in target columns (sbp, dbp, map) for modeling-eligible windows
    target_cols = ["sbp", "dbp", "map"]
    for col in target_cols:
        if col in eligible_df.columns:
            nan_count = eligible_df[col].isna().sum()
            inf_count = np.isinf(eligible_df[col]).sum()
            if nan_count > 0 or inf_count > 0:
                raise AssertionError(
                    f"CHECK 8 FAILED: Modeling-eligible column '{col}' contains "
                    f"NaN={nan_count}, Inf={inf_count}!"
                )
    logger.info("  ✓ Check 8 Passed: No NaN/Inf in target columns for modeling-eligible windows.")

    # Check 9: No ECG-derived column exists in the Phase 2 modeling manifest
    forbidden_ecg_substrings = ["ecg", "ptt", "pat", "qrs", "pr_interval", "qt_interval"]
    found_ecg_cols = [
        col for col in window_manifest_df.columns
        if any(sub in col.lower() for sub in forbidden_ecg_substrings)
    ]
    if len(found_ecg_cols) > 0:
        raise AssertionError(f"CHECK 9 FAILED: ECG-derived columns found in manifest: {found_ecg_cols}!")
    logger.info("  ✓ Check 9 Passed: No ECG-derived columns present in modeling manifest.")

    summary = {
        "all_checks_passed": True,
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "train_records_count": len(train_records),
        "val_records_count": len(val_records),
        "test_records_count": len(test_records),
        "eligible_windows_count": len(eligible_df),
        "leakage_control_note": LEAKAGE_CONTROL_NOTE,
    }
    return summary


def compute_split_distribution_summary(
    window_manifest_df: pd.DataFrame,
    record_split_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Computes window counts, eligibility rates, and target distributions (SBP, DBP, MAP, HR)
    stratified by train, validation, and test partitions.
    """
    # Merge split assignment onto window manifest if not already present
    if "split" not in window_manifest_df.columns:
        df = window_manifest_df.merge(
            record_split_df[["record_id", "split"]],
            on="record_id",
            how="left",
        )
    else:
        df = window_manifest_df.copy()

    split_summary = {}
    for split_name in ["train", "val", "test"]:
        sub_df = df[df["split"] == split_name]
        elig_df = sub_df[sub_df["modeling_eligible"] == True]

        stats = {
            "num_records": int(sub_df["record_id"].nunique()),
            "total_windows": int(len(sub_df)),
            "ppg_valid_windows": int((sub_df["ppg_valid"] == True).sum()),
            "abp_valid_windows": int((sub_df["abp_valid"] == True).sum()),
            "eligible_windows": int(len(elig_df)),
            "eligibility_rate_pct": float(len(elig_df) / max(1, len(sub_df)) * 100.0),
            "sbp": {
                "mean": float(elig_df["sbp"].mean()) if len(elig_df) else np.nan,
                "std": float(elig_df["sbp"].std()) if len(elig_df) else np.nan,
                "min": float(elig_df["sbp"].min()) if len(elig_df) else np.nan,
                "p25": float(elig_df["sbp"].quantile(0.25)) if len(elig_df) else np.nan,
                "median": float(elig_df["sbp"].median()) if len(elig_df) else np.nan,
                "p75": float(elig_df["sbp"].quantile(0.75)) if len(elig_df) else np.nan,
                "max": float(elig_df["sbp"].max()) if len(elig_df) else np.nan,
            },
            "dbp": {
                "mean": float(elig_df["dbp"].mean()) if len(elig_df) else np.nan,
                "std": float(elig_df["dbp"].std()) if len(elig_df) else np.nan,
                "min": float(elig_df["dbp"].min()) if len(elig_df) else np.nan,
                "p25": float(elig_df["dbp"].quantile(0.25)) if len(elig_df) else np.nan,
                "median": float(elig_df["dbp"].median()) if len(elig_df) else np.nan,
                "p75": float(elig_df["dbp"].quantile(0.75)) if len(elig_df) else np.nan,
                "max": float(elig_df["dbp"].max()) if len(elig_df) else np.nan,
            },
            "map": {
                "mean": float(elig_df["map"].mean()) if len(elig_df) else np.nan,
                "std": float(elig_df["map"].std()) if len(elig_df) else np.nan,
                "min": float(elig_df["map"].min()) if len(elig_df) else np.nan,
                "p25": float(elig_df["map"].quantile(0.25)) if len(elig_df) else np.nan,
                "median": float(elig_df["map"].median()) if len(elig_df) else np.nan,
                "p75": float(elig_df["map"].quantile(0.75)) if len(elig_df) else np.nan,
                "max": float(elig_df["map"].max()) if len(elig_df) else np.nan,
            },
            "estimated_hr_bpm": {
                "mean": float(elig_df["estimated_hr_bpm"].mean()) if len(elig_df) else np.nan,
                "std": float(elig_df["estimated_hr_bpm"].std()) if len(elig_df) else np.nan,
                "median": float(elig_df["estimated_hr_bpm"].median()) if len(elig_df) else np.nan,
            },
        }
        split_summary[split_name] = stats

    return split_summary
