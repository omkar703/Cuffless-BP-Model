"""
Dataset Quality Statistics & Diagnostic Profiling Module.

Aggregates record-level manifest data into global distribution statistics,
validity ratios, physiological summaries, and artifact diagnostics.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_dataset_qc_statistics(df_manifest: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive dataset quality control and physiological statistics
    from the QC manifest DataFrame.
    """
    total_records = len(df_manifest)
    if total_records == 0:
        return {"total_records": 0}

    # Decoupled validity counts
    structural_valid_count = int(df_manifest["structural_valid"].sum())
    ppg_valid_count = int(df_manifest["ppg_valid"].sum())
    abp_valid_count = int(df_manifest["abp_valid"].sum())
    paired_valid_count = int(df_manifest["paired_valid"].sum())

    status_counts = df_manifest["quality_status"].value_counts().to_dict()

    # Duration and length metrics
    durations = df_manifest["duration_seconds"].values
    num_samples = df_manifest["num_samples"].values

    duration_stats = {
        "min_sec": float(np.min(durations)),
        "max_sec": float(np.max(durations)),
        "mean_sec": float(np.mean(durations)),
        "median_sec": float(np.median(durations)),
        "std_sec": float(np.std(durations)),
        "p25_sec": float(np.percentile(durations, 25)),
        "p75_sec": float(np.percentile(durations, 75)),
    }

    samples_stats = {
        "min_samples": int(np.min(num_samples)),
        "max_samples": int(np.max(num_samples)),
        "mean_samples": float(np.mean(num_samples)),
        "median_samples": float(np.median(num_samples)),
    }

    # PPG Diagnostics (computed on PPG-valid records)
    df_ppg = df_manifest[df_manifest["ppg_valid"]]
    ppg_stats = {}
    if len(df_ppg) > 0:
        ppg_ptps = df_ppg["ppg_ptp"].dropna().values
        ppg_stds = df_ppg["ppg_std"].dropna().values
        ppg_stats = {
            "valid_records_count": len(df_ppg),
            "valid_percentage": float(len(df_ppg) / total_records * 100.0),
            "ptp_mean": float(np.mean(ppg_ptps)),
            "ptp_median": float(np.median(ppg_ptps)),
            "ptp_min": float(np.min(ppg_ptps)),
            "ptp_max": float(np.max(ppg_ptps)),
            "ptp_p05": float(np.percentile(ppg_ptps, 5)),
            "ptp_p95": float(np.percentile(ppg_ptps, 95)),
            "std_mean": float(np.mean(ppg_stds)),
            "std_median": float(np.median(ppg_stds)),
        }

    # ABP Reference Diagnostics (computed on ABP-valid records)
    df_abp = df_manifest[df_manifest["abp_valid"]]
    abp_stats = {}
    if len(df_abp) > 0:
        sbps = df_abp["sbp_mean"].dropna().values
        dbps = df_abp["dbp_mean"].dropna().values
        maps = df_abp["map_mean"].dropna().values
        hrs = df_abp["abp_estimated_hr_bpm"].dropna().values
        beat_pcts = df_abp["abp_valid_beat_pct"].dropna().values

        abp_stats = {
            "valid_records_count": len(df_abp),
            "valid_percentage": float(len(df_abp) / total_records * 100.0),
            "sbp_mean": float(np.mean(sbps)) if len(sbps) > 0 else np.nan,
            "sbp_std": float(np.std(sbps)) if len(sbps) > 0 else np.nan,
            "sbp_median": float(np.median(sbps)) if len(sbps) > 0 else np.nan,
            "sbp_min": float(np.min(sbps)) if len(sbps) > 0 else np.nan,
            "sbp_max": float(np.max(sbps)) if len(sbps) > 0 else np.nan,
            "dbp_mean": float(np.mean(dbps)) if len(dbps) > 0 else np.nan,
            "dbp_std": float(np.std(dbps)) if len(dbps) > 0 else np.nan,
            "dbp_median": float(np.median(dbps)) if len(dbps) > 0 else np.nan,
            "dbp_min": float(np.min(dbps)) if len(dbps) > 0 else np.nan,
            "dbp_max": float(np.max(dbps)) if len(dbps) > 0 else np.nan,
            "map_mean": float(np.mean(maps)) if len(maps) > 0 else np.nan,
            "map_std": float(np.std(maps)) if len(maps) > 0 else np.nan,
            "hr_mean": float(np.mean(hrs)) if len(hrs) > 0 else np.nan,
            "hr_std": float(np.std(hrs)) if len(hrs) > 0 else np.nan,
            "beat_valid_pct_mean": float(np.mean(beat_pcts)) if len(beat_pcts) > 0 else np.nan,
        }

    # Rejection reasons breakdown
    ppg_reject_reasons = (
        df_manifest[~df_manifest["ppg_valid"]]["ppg_rejection_reason"]
        .value_counts()
        .to_dict()
    )
    abp_reject_reasons = (
        df_manifest[~df_manifest["abp_valid"]]["abp_rejection_reason"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_records": total_records,
        "structural_valid_count": structural_valid_count,
        "ppg_valid_count": ppg_valid_count,
        "ppg_valid_percentage": float(ppg_valid_count / total_records * 100.0),
        "abp_valid_count": abp_valid_count,
        "abp_valid_percentage": float(abp_valid_count / total_records * 100.0),
        "paired_valid_count": paired_valid_count,
        "paired_valid_percentage": float(paired_valid_count / total_records * 100.0),
        "quality_status_breakdown": status_counts,
        "duration_statistics": duration_stats,
        "samples_statistics": samples_stats,
        "ppg_signal_diagnostics": ppg_stats,
        "abp_reference_diagnostics": abp_stats,
        "ppg_rejection_reasons": ppg_reject_reasons,
        "abp_rejection_reasons": abp_reject_reasons,
    }
