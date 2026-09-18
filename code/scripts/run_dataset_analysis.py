#!/usr/bin/env python3
"""
CLI Runner Script for Phase 1: PPG-Only Blood Pressure Dataset Analysis & Quality Control.

Usage:
    python scripts/run_dataset_analysis.py              # Full run on all 12 MAT parts
    python scripts/run_dataset_analysis.py --max-parts 2 # Test run on first 2 parts
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Ensure 'code' root directory is on PYTHONPATH
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config.config import (
    DATASET_DIR,
    OUTPUT_DIR,
    STATISTICS_DIR,
    FIGURES_DIR,
    QC_MANIFEST_FILENAME,
    QC_SUMMARY_JSON,
    QC_REPORT_FILENAME,
    REPRESENTATIVE_RECORD_IDS,
)
from utils.logging_utils import setup_logger
from data.loader import get_mat_files, record_stream_generator
from data.quality_control import perform_record_quality_control
from data.dataset_statistics import compute_dataset_qc_statistics
from visualization.signal_plots import generate_representative_signal_plots
from visualization.dataset_plots import generate_all_dataset_distribution_plots

logger = setup_logger("cli_dataset_analysis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Dataset Cleaning, Validation & Visualization CLI Runner"
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Maximum number of MAT parts to process (default: all available parts)",
    )
    parser.add_argument(
        "--num-vis-records",
        type=int,
        default=6,
        help="Number of representative records to visualize (default: 6)",
    )
    return parser.parse_args()


def generate_markdown_report(stats: dict, manifest_path: Path, figures: list, output_report_path: Path):
    """Writes the comprehensive DATASET_CLEANING_REPORT.md."""
    dur_stats = stats.get("duration_statistics", {})
    ppg_stats = stats.get("ppg_signal_diagnostics", {})
    abp_stats = stats.get("abp_reference_diagnostics", {})
    breakdown = stats.get("quality_status_breakdown", {})
    ppg_reasons = stats.get("ppg_rejection_reasons", {})
    abp_reasons = stats.get("abp_rejection_reasons", {})

    md_content = f"""# Phase 1: Dataset Quality Control & Validation Audit Report

## 1. Executive Dataset Summary

- **Dataset Source**: PhysioNet MIMIC-II Waveform Database (Curated by Kachuee et al.)
- **Acquisition Sampling Rate**: 125 Hz
- **Total Records Analyzed**: {stats.get('total_records', 0):,}
- **Structurally Intact Records**: {stats.get('structural_valid_count', 0):,} ({stats.get('structural_valid_count', 0)/max(1, stats.get('total_records', 1))*100:.1f}%)
- **Valid Paired Records (PPG + ABP)**: {stats.get('paired_valid_count', 0):,} ({stats.get('paired_valid_percentage', 0.0):.1f}%)
- **Valid PPG-Only Records**: {breakdown.get('valid_ppg_only', 0):,} ({breakdown.get('valid_ppg_only', 0)/max(1, stats.get('total_records', 1))*100:.1f}%)
- **Valid ABP-Only Records**: {breakdown.get('valid_abp_only', 0):,} ({breakdown.get('valid_abp_only', 0)/max(1, stats.get('total_records', 1))*100:.1f}%)
- **Completely Rejected Records**: {breakdown.get('rejected_both', 0):,} ({breakdown.get('rejected_both', 0)/max(1, stats.get('total_records', 1))*100:.1f}%)

> [!NOTE]
> **Decoupled Quality Gates**: Records with corrupted arterial blood pressure (catheter disconnects, clamping, line damping) are **not discarded from PPG modeling**. Their PPG signals remain preserved and labeled as `valid_ppg_only`.

---

## 2. Raw Dataset Duration & Length Statistics

| Metric | Duration (Seconds) | Duration (Minutes) | Samples Count |
| :--- | :---: | :---: | :---: |
| **Minimum** | {dur_stats.get('min_sec', 0.0):.1f} s | {dur_stats.get('min_sec', 0.0)/60.0:.2f} min | {stats.get('samples_statistics', {}).get('min_samples', 0):,} |
| **Maximum** | {dur_stats.get('max_sec', 0.0):.1f} s | {dur_stats.get('max_sec', 0.0)/60.0:.2f} min | {stats.get('samples_statistics', {}).get('max_samples', 0):,} |
| **Mean** | {dur_stats.get('mean_sec', 0.0):.1f} s | {dur_stats.get('mean_sec', 0.0)/60.0:.2f} min | {stats.get('samples_statistics', {}).get('mean_samples', 0):,.0f} |
| **Median** | {dur_stats.get('median_sec', 0.0):.1f} s | {dur_stats.get('median_sec', 0.0)/60.0:.2f} min | {stats.get('samples_statistics', {}).get('median_samples', 0):,.0f} |
| **Standard Deviation** | {dur_stats.get('std_sec', 0.0):.1f} s | {dur_stats.get('std_sec', 0.0)/60.0:.2f} min | — |

---

## 3. PPG Signal Characteristics & Diagnostics

- **Valid PPG Count**: {ppg_stats.get('valid_records_count', 0):,} ({ppg_stats.get('valid_percentage', 0.0):.1f}%)
- **Peak-to-Peak (PTP) Amplitude Mean**: {ppg_stats.get('ptp_mean', 0.0):.3f} V
- **Peak-to-Peak (PTP) Amplitude Median**: {ppg_stats.get('ptp_median', 0.0):.3f} V
- **PTP Amplitude 5th – 95th Percentile**: [{ppg_stats.get('ptp_p05', 0.0):.3f} V, {ppg_stats.get('ptp_p95', 0.0):.3f} V]
- **Signal Standard Deviation (Mean)**: {ppg_stats.get('std_mean', 0.0):.4f}

### Main Quality Control Findings for PPG:
1. **Flatline Dropouts**: Sensor detachment manifests as $\sigma < 10^{{-5}}$ and zero variance.
2. **Clipping / Saturation**: Certain sensors hit ADC rails where $>5\%$ of samples clamp to the extrema.
3. **Amplitude Variation**: Absolute optical amplitude varies heavily by subject tissue and probe gain; relative contour features are required rather than absolute thresholding.

---

## 4. ABP Reference Signal Characteristics & Beat Diagnostics

- **Valid ABP Reference Count**: {abp_stats.get('valid_records_count', 0):,} ({abp_stats.get('valid_percentage', 0.0):.1f}%)
- **Mean SBP (Beat-by-Beat Peak)**: {abp_stats.get('sbp_mean', 0.0):.1f} ± {abp_stats.get('sbp_std', 0.0):.1f} mmHg
- **Median SBP**: {abp_stats.get('sbp_median', 0.0):.1f} mmHg (Range: {abp_stats.get('sbp_min', 0.0):.1f} – {abp_stats.get('sbp_max', 0.0):.1f} mmHg)
- **Mean DBP (Beat-by-Beat Trough)**: {abp_stats.get('dbp_mean', 0.0):.1f} ± {abp_stats.get('dbp_std', 0.0):.1f} mmHg
- **Median DBP**: {abp_stats.get('dbp_median', 0.0):.1f} mmHg (Range: {abp_stats.get('dbp_min', 0.0):.1f} – {abp_stats.get('dbp_max', 0.0):.1f} mmHg)
- **Mean Arterial Pressure (MAP)**: {abp_stats.get('map_mean', 0.0):.1f} ± {abp_stats.get('map_std', 0.0):.1f} mmHg
- **Estimated Heart Rate**: {abp_stats.get('hr_mean', 0.0):.1f} ± {abp_stats.get('hr_std', 0.0):.1f} bpm
- **Average Valid Beat Percentage**: {abp_stats.get('beat_valid_pct_mean', 0.0):.1f}%

---

## 5. Signal Rejection Reasons Breakdown

### PPG Rejection Causes:
| Rejection Reason | Occurrences |
| :--- | :---: |
"""
    for reason, count in ppg_reasons.items():
        md_content += f"| `{reason}` | {count:,} |\n"

    md_content += """
### ABP Rejection Causes:
| Rejection Reason | Occurrences |
| :--- | :---: |
"""
    for reason, count in abp_reasons.items():
        md_content += f"| `{reason}` | {count:,} |\n"

    md_content += f"""
---

## 6. Generated Visualizations & Artifacts

All diagnostic figures have been generated and saved under `code/outputs/figures/`:
1. `01_record_duration_distribution.png`: Histogram and KDE of recording durations.
2. `02_ppg_amplitude_distribution.png`: Distribution of PPG peak-to-peak amplitudes.
3. `03_abp_continuous_distribution.png`: Distribution of ABP continuous integration.
4. `04_sbp_distribution.png`: Distribution of beat-by-beat SBP ground truth.
5. `05_dbp_distribution.png`: Distribution of beat-by-beat DBP ground truth.
6. `06_quality_status_breakdown.png`: Bar chart of decoupled quality categories.
7. `07_rejection_reasons_breakdown.png`: Pareto charts of anomaly causes.

---

## 7. Strategic Decisions for Phase 2 Modeling

1. **Strictly Calibration-Free**: Avoid the calibrated delta crutch identified in earlier work.
2. **Pure PPG Pipeline**: Discard all ECG channels and PTT dependencies.
3. **Record-Level Partitioning**: All training/validation/test splits in Phase 2 must use the persistent `record_id` from `{manifest_path.name}` to ensure zero patient leakage.
4. **Offline Filter Transition**: The 0.5–8.0 Hz zero-phase filter applied here serves as the ground-truth offline standard. Phase 2 must design an equivalent causal streaming filter suitable for ESP32 deployment.
"""

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Report saved to {output_report_path}")


def main():
    args = parse_args()
    start_time = time.time()
    logger.info("Starting Phase 1 Dataset Quality Control & Validation Analysis...")

    mat_files = get_mat_files()
    if not mat_files:
        logger.error(f"No part_*.mat files found in {DATASET_DIR}!")
        sys.exit(1)

    if args.max_parts:
        mat_files = mat_files[:args.max_parts]
        logger.info(f"Limiting execution to first {len(mat_files)} MAT parts.")
    else:
        logger.info(f"Processing all {len(mat_files)} MAT parts.")

    # 1. Stream records and compute QC diagnostics
    manifest_rows = []
    representative_records = []
    num_to_collect = args.num_vis_records

    logger.info("Executing record-level quality control pipeline...")
    for rec in tqdm(record_stream_generator(mat_files), desc="Auditing Records"):
        qc_result = perform_record_quality_control(rec)
        manifest_rows.append(qc_result)

        # Collect representative samples for visual inspection
        if len(representative_records) < num_to_collect:
            if qc_result["paired_valid"] or len(manifest_rows) % 100 == 0:
                representative_records.append(rec)

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_csv_path = STATISTICS_DIR / QC_MANIFEST_FILENAME
    df_manifest.to_csv(manifest_csv_path, index=False)
    logger.info(f"Quality Control Manifest generated with {len(df_manifest):,} records at {manifest_csv_path}")

    # 2. Compute Aggregate Summary Statistics
    logger.info("Computing global quality statistics...")
    stats = compute_dataset_qc_statistics(df_manifest)

    summary_json_path = STATISTICS_DIR / QC_SUMMARY_JSON
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Summary statistics exported to {summary_json_path}")

    # 3. Generate Visualizations
    logger.info("Generating dataset-level distribution plots...")
    dist_figures = generate_all_dataset_distribution_plots(df_manifest, save_dir=FIGURES_DIR)

    logger.info(f"Generating representative waveform plots for {len(representative_records)} records...")
    sig_figures = generate_representative_signal_plots(representative_records, save_dir=FIGURES_DIR, window_sec=10.0)

    # 4. Generate Final Audit Report
    report_path = STATISTICS_DIR / QC_REPORT_FILENAME
    generate_markdown_report(stats, manifest_csv_path, dist_figures + sig_figures, report_path)

    elapsed = time.time() - start_time
    logger.info(f"Phase 1 Dataset Analysis completed in {elapsed:.1f} seconds.")
    print(f"\n✅ Quality Control Manifest: {manifest_csv_path}")
    print(f"✅ Full Audit Report: {report_path}")
    print(f"✅ Visual Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
