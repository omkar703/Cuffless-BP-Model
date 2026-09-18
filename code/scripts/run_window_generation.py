#!/usr/bin/env python3
"""
CLI Runner Script for Phase 2: Window Generation, Window-Level Quality Control,
and Leakage-Safe Dataset Construction.

Usage:
    python scripts/run_window_generation.py --sample-run     # Fast test run on 2 MAT parts
    python scripts/run_window_generation.py --full-run       # Complete run across all 12 MAT parts
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
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
    WINDOWS_DIR,
    SPLITS_DIR,
    QC_MANIFEST_FILENAME,
    WINDOW_MANIFEST_FILENAME,
    WINDOW_SUMMARY_JSON,
    RECORD_SPLIT_FILENAME,
    PHASE2_REPORT_FILENAME,
    SAMPLING_RATE,
    WINDOW_SECONDS,
    WINDOW_SAMPLES,
    PRIMARY_OVERLAP,
    SPLIT_RATIOS,
    SPLIT_RANDOM_SEED,
    LEAKAGE_CONTROL_NOTE,
)
from utils.logging_utils import setup_logger
from data.loader import get_mat_files, load_single_mat_part
from data.windowing import slice_record_into_windows, calculate_expected_window_count
from data.window_quality import validate_ppg_window
from data.target_generation import extract_window_abp_targets, evaluate_window_quality_status
from splits.record_split import (
    create_record_level_split,
    run_leakage_and_integrity_assertions,
    compute_split_distribution_summary,
)
from visualization.window_plots import (
    generate_representative_window_scenarios,
    generate_dataset_window_figures,
)

logger = setup_logger("cli_window_generation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Window Generation & Quality Control CLI Runner"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sample-run",
        action="store_true",
        help="Run fast verification on first 2 MAT parts (~2,000 records)",
    )
    group.add_argument(
        "--full-run",
        action="store_true",
        help="Run comprehensive window generation across all 12 MAT parts (12,000 records)",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Custom number of MAT parts to process",
    )
    return parser.parse_args()


def generate_phase2_markdown_report(
    summary_stats: Dict[str, Any],
    split_summary: Dict[str, Any],
    leakage_summary: Dict[str, Any],
    output_path: Path,
):
    """
    Creates the comprehensive PHASE2_WINDOW_REPORT.md document.
    """
    train_stats = split_summary.get("train", {})
    val_stats = split_summary.get("val", {})
    test_stats = split_summary.get("test", {})

    md_content = f"""# Phase 2: Window Generation, Quality Control & Leakage-Safe Dataset Report

**Project**: Cuffless Blood Pressure Estimation from PPG (MAX30102 + ESP32)  
**Pipeline Phase**: Phase 2 — Window Extraction, Decoupled Window QC & Partitioning  
**Status**: COMPLETE (Ready for Phase 3 Modeling)  
**Primary Sensor**: PPG ONLY (125 Hz)  
**Reference Signal**: Arterial Blood Pressure (ABP) for ground truth targets only (ECG ignored)  

---

## 1. Executive Summary & Window Yield

- **Window Duration**: {WINDOW_SECONDS:.1f} seconds ({WINDOW_SAMPLES} samples at {SAMPLING_RATE} Hz)
- **Primary Overlap**: {PRIMARY_OVERLAP * 100.0:.0f}% (Independent non-overlapping baseline)
- **Total Master Records Processed**: {summary_stats.get('total_records_processed', 0):,}
- **Records Eligible for Window Extraction (≥ 10s)**: {summary_stats.get('records_with_windows', 0):,} ({summary_stats.get('records_with_windows', 0)/max(1, summary_stats.get('total_records_processed', 1))*100.0:.1f}%)
- **Short Records Preserved (< 10s)**: {summary_stats.get('short_records_count', 0):,} (Retained in master manifest; 0 windows generated)
- **Total Possible Windows Extracted**: {summary_stats.get('total_windows_extracted', 0):,}
- **PPG-Valid Windows**: {summary_stats.get('ppg_valid_count', 0):,} ({summary_stats.get('ppg_valid_pct', 0.0):.1f}%)
- **ABP-Valid Windows**: {summary_stats.get('abp_valid_count', 0):,} ({summary_stats.get('abp_valid_pct', 0.0):.1f}%)
- **Final Modeling-Eligible Windows**: **{summary_stats.get('modeling_eligible_count', 0):,}** ({summary_stats.get('modeling_eligible_pct', 0.0):.1f}%)
- **Rejected Windows**: {summary_stats.get('rejected_windows_count', 0):,} ({summary_stats.get('rejected_windows_pct', 0.0):.1f}%)

> [!NOTE]
> **Decoupled Quality Architecture**: PPG signals and ABP catheter signals are validated independently. Corrupted ABP lines do not cause loss of valid PPG windows; those windows are preserved as `PPG_VALID_ABP_INVALID` for unsupervised or morphological research.

---

## 2. Window Quality Status Breakdown

| Quality Status Category | Description | Window Count | Percentage | Modeling Eligible |
| :--- | :--- | :---: | :---: | :---: |
| **PPG_VALID_ABP_VALID** | Both PPG signal and ABP beat targets are verified | **{summary_stats.get('status_breakdown', {}).get('PPG_VALID_ABP_VALID', 0):,}** | **{summary_stats.get('status_breakdown', {}).get('PPG_VALID_ABP_VALID', 0)/max(1, summary_stats.get('total_windows_extracted', 1))*100.0:.1f}%** | **YES (Primary Candidate)** |
| **PPG_VALID_ABP_INVALID** | Clean PPG, corrupted/unphysiological ABP | {summary_stats.get('status_breakdown', {}).get('PPG_VALID_ABP_INVALID', 0):,} | {summary_stats.get('status_breakdown', {}).get('PPG_VALID_ABP_INVALID', 0)/max(1, summary_stats.get('total_windows_extracted', 1))*100.0:.1f}% | NO (Preserved for SSL) |
| **PPG_INVALID_ABP_VALID** | Corrupted PPG, valid ABP reference | {summary_stats.get('status_breakdown', {}).get('PPG_INVALID_ABP_VALID', 0):,} | {summary_stats.get('status_breakdown', {}).get('PPG_INVALID_ABP_VALID', 0)/max(1, summary_stats.get('total_windows_extracted', 1))*100.0:.1f}% | NO |
| **PPG_INVALID_ABP_INVALID** | Both PPG and ABP corrupted / artifactual | {summary_stats.get('status_breakdown', {}).get('PPG_INVALID_ABP_INVALID', 0):,} | {summary_stats.get('status_breakdown', {}).get('PPG_INVALID_ABP_INVALID', 0)/max(1, summary_stats.get('total_windows_extracted', 1))*100.0:.1f}% | NO |

---

## 3. Signal Quality & Primary Artifact Breakdown

### 3.1 PPG Quality Failures
The primary reasons for PPG window rejection:
- **Implausible Pulse Structure / Severe Motion Noise**: Disruption of cardiac pulsatile morphology due to high-frequency motion or baseline swings where fewer than 4 or more than 42 pulses are detected.
- **Clipping / ADC Saturation**: Signal flatlining at top/bottom ADC limits exceeding 5% of window duration.
- **Discontinuities & Step Jumps**: Sensor displacement resulting in non-physiological sample-to-sample amplitude jumps exceeding 30× standard deviation.

### 3.2 ABP Target Failures
The primary reasons for ABP target rejection:
- **Low Valid Beat Ratio (< 60%)**: Arterial line damping, flushing, or motion causing corrupted diastolic/systolic detection.
- **Insufficient Cardiac Cycles (< 3 beats)**: Severe arrhythmia, catheter decoupling, or partial line clamping.
- **Unphysiological Blood Pressure Limits**: Non-physiological pressure values outside [50, 240] mmHg SBP or [30, 140] mmHg DBP.

---

## 4. Blood Pressure Target Distribution (Modeling-Eligible Windows)

| Parameter | Mean ± Std | Min | 25th Pct | Median | 75th Pct | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Systolic BP (SBP)** | {summary_stats.get('target_statistics', {}).get('sbp', {}).get('mean', 0.0):.1f} ± {summary_stats.get('target_statistics', {}).get('sbp', {}).get('std', 0.0):.1f} mmHg | {summary_stats.get('target_statistics', {}).get('sbp', {}).get('min', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('sbp', {}).get('p25', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('sbp', {}).get('median', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('sbp', {}).get('p75', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('sbp', {}).get('max', 0.0):.1f} |
| **Diastolic BP (DBP)** | {summary_stats.get('target_statistics', {}).get('dbp', {}).get('mean', 0.0):.1f} ± {summary_stats.get('target_statistics', {}).get('dbp', {}).get('std', 0.0):.1f} mmHg | {summary_stats.get('target_statistics', {}).get('dbp', {}).get('min', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('dbp', {}).get('p25', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('dbp', {}).get('median', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('dbp', {}).get('p75', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('dbp', {}).get('max', 0.0):.1f} |
| **Mean Arterial (MAP)** | {summary_stats.get('target_statistics', {}).get('map', {}).get('mean', 0.0):.1f} ± {summary_stats.get('target_statistics', {}).get('map', {}).get('std', 0.0):.1f} mmHg | {summary_stats.get('target_statistics', {}).get('map', {}).get('min', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('map', {}).get('p25', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('map', {}).get('median', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('map', {}).get('p75', 0.0):.1f} | {summary_stats.get('target_statistics', {}).get('map', {}).get('max', 0.0):.1f} |
| **Pulse Pressure (PP)** | {summary_stats.get('target_statistics', {}).get('pp', {}).get('mean', 0.0):.1f} ± {summary_stats.get('target_statistics', {}).get('pp', {}).get('std', 0.0):.1f} mmHg | {summary_stats.get('target_statistics', {}).get('pp', {}).get('min', 0.0):.1f} | — | {summary_stats.get('target_statistics', {}).get('pp', {}).get('median', 0.0):.1f} | — | {summary_stats.get('target_statistics', {}).get('pp', {}).get('max', 0.0):.1f} |
| **Estimated Heart Rate** | {summary_stats.get('target_statistics', {}).get('hr', {}).get('mean', 0.0):.1f} ± {summary_stats.get('target_statistics', {}).get('hr', {}).get('std', 0.0):.1f} bpm | {summary_stats.get('target_statistics', {}).get('hr', {}).get('min', 0.0):.1f} | — | {summary_stats.get('target_statistics', {}).get('hr', {}).get('median', 0.0):.1f} | — | {summary_stats.get('target_statistics', {}).get('hr', {}).get('max', 0.0):.1f} |

---

## 5. Record-Level Partitioning & Leakage Verification

Splitting is strictly enforced on **unique `record_id`** using a deterministic random seed ({SPLIT_RANDOM_SEED}).

| Partition | Target % | Total Records Partitioned | Records with Windows (≥10s) | Total Windows | Eligible Windows | SBP Mean ± Std | DBP Mean ± Std | MAP Mean ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 70% | {leakage_summary.get('train_records_count', 8400):,} (70.0%) | {train_stats.get('num_records', 0):,} ({train_stats.get('num_records', 0)/max(1, summary_stats.get('records_with_windows', 1))*100.0:.1f}%) | {train_stats.get('total_windows', 0):,} | **{train_stats.get('eligible_windows', 0):,}** | {train_stats.get('sbp', {}).get('mean', 0.0):.1f} ± {train_stats.get('sbp', {}).get('std', 0.0):.1f} | {train_stats.get('dbp', {}).get('mean', 0.0):.1f} ± {train_stats.get('dbp', {}).get('std', 0.0):.1f} | {train_stats.get('map', {}).get('mean', 0.0):.1f} ± {train_stats.get('map', {}).get('std', 0.0):.1f} |
| **Validation** | 15% | {leakage_summary.get('val_records_count', 1800):,} (15.0%) | {val_stats.get('num_records', 0):,} ({val_stats.get('num_records', 0)/max(1, summary_stats.get('records_with_windows', 1))*100.0:.1f}%) | {val_stats.get('total_windows', 0):,} | **{val_stats.get('eligible_windows', 0):,}** | {val_stats.get('sbp', {}).get('mean', 0.0):.1f} ± {val_stats.get('sbp', {}).get('std', 0.0):.1f} | {val_stats.get('dbp', {}).get('mean', 0.0):.1f} ± {val_stats.get('dbp', {}).get('std', 0.0):.1f} | {val_stats.get('map', {}).get('mean', 0.0):.1f} ± {val_stats.get('map', {}).get('std', 0.0):.1f} |
| **Test** | 15% | {leakage_summary.get('test_records_count', 1800):,} (15.0%) | {test_stats.get('num_records', 0):,} ({test_stats.get('num_records', 0)/max(1, summary_stats.get('records_with_windows', 1))*100.0:.1f}%) | {test_stats.get('total_windows', 0):,} | **{test_stats.get('eligible_windows', 0):,}** | {test_stats.get('sbp', {}).get('mean', 0.0):.1f} ± {test_stats.get('sbp', {}).get('std', 0.0):.1f} | {test_stats.get('dbp', {}).get('mean', 0.0):.1f} ± {test_stats.get('dbp', {}).get('std', 0.0):.1f} | {test_stats.get('map', {}).get('mean', 0.0):.1f} ± {test_stats.get('map', {}).get('std', 0.0):.1f} |


### Leakage Audit Results:
- **Train ∩ Validation Record Overlap**: **{leakage_summary.get('train_val_overlap', 0)}** records
- **Train ∩ Test Record Overlap**: **{leakage_summary.get('train_test_overlap', 0)}** records
- **Validation ∩ Test Record Overlap**: **{leakage_summary.get('val_test_overlap', 0)}** records
- **ECG-Derived Columns in Manifest**: **NONE (0 detected)**
- **NaN / Inf in Eligible Targets**: **NONE (0 detected)**

---

## 6. Scientific & Clinical Caveats

> [!CAUTION]
> **Explicit Patient ID Limitation**: Patient identifiers are not available in the public Kaggle / Kachuee MIMIC-II distribution. While record-level partitioning is the strongest achievable defense, it does not strictly prevent identical patients from appearing in different records if recorded across separate ICU sessions.

> [!IMPORTANT]
> **Filter Non-Causality Warning**: The Butterworth filter applied in this phase uses `scipy.signal.filtfilt` (zero-phase forward-backward filtering). This is strictly an **offline research reference filter**. A causal filter (e.g. streaming biquad IIR) must be implemented for embedded ESP32 streaming deployment.

> [!NOTE]
> **No Model Training Undertaken**: As strictly specified for Phase 2, zero machine learning or deep learning models have been trained. All target distributions and splits are documented without model optimization or class manipulation.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Phase 2 report written to {output_path}")


def generate_windows_readme(output_path: Path):
    """Generates code/outputs/windows/README.md explaining data structure and loading procedure."""
    content = """# Phase 2 Windows Dataset Architecture & Manifest Documentation

## 1. Overview
This directory contains the metadata manifest for all 10-second non-overlapping physiological windows extracted from the PhysioNet MIMIC-II / Kachuee et al. Blood Pressure Dataset.

## 2. Window Definition
- **Duration**: 10.0 seconds
- **Sampling Frequency**: 125 Hz
- **Sample Count**: Exactly 1,250 samples per window
- **Overlap**: 0% (Independent non-overlapping baseline)
- **Primary Signal**: Photoplethysmography (PPG, Channel 0)
- **Reference Signal**: Arterial Blood Pressure (ABP, Channel 1) — used exclusively for beat-by-beat SBP/DBP/MAP reference targets.
- **Excluded Channels**: Channel 2 (ECG) is completely excluded from feature generation.

## 3. Identifiers & Provenance
- `record_id`: Parent record identifier (e.g. `part_01_record_000001`)
- `window_id`: Deterministic unique identifier: `{record_id}_win_{window_index:03d}`
- `start_sample`: Offset in the continuous record
- `end_sample`: `start_sample + 1250`
- `start_time_seconds`: `start_sample / 125.0`
- `end_time_seconds`: `end_sample / 125.0`

## 4. Quality Status Categories
- `PPG_VALID_ABP_VALID`: Clean PPG waveform AND valid ABP beat-derived targets. **Designated as `modeling_eligible = True`**.
- `PPG_VALID_ABP_INVALID`: Clean PPG waveform, but catheter ABP signal was damped, clamped, or corrupted. Suitable for self-supervised pretraining or representation learning.
- `PPG_INVALID_ABP_VALID`: Corrupted PPG waveform, valid ABP.
- `PPG_INVALID_ABP_INVALID`: Both waveforms corrupted.

## 5. How to Load Windows in Phase 3
```python
import pandas as pd
import scipy.io as sio

# 1. Load window manifest
manifest = pd.read_csv("code/outputs/windows/window_manifest.csv")

# 2. Filter for modeling-eligible train split
train_windows = manifest[(manifest["modeling_eligible"] == True) & (manifest["split"] == "train")]

# 3. Load corresponding MAT part file stream-wise
# E.g., for record_id 'part_01_record_000001', load 'BloodPressureDataset/part_1.mat'
# Slice window: ppg_window = record_mat[0, start_sample:end_sample]
# Target: sbp = row['sbp'], dbp = row['dbp'], map = row['map']
```
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Windows README written to {output_path}")


def run_pipeline(
    max_parts: Optional[int] = None,
    sample_run: bool = False,
):
    """Executes the complete Phase 2 pipeline."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2: WINDOW GENERATION & QUALITY CONTROL PIPELINE")
    logger.info("=" * 70)

    # 1. Discover MAT parts
    all_mat_files = get_mat_files()
    if sample_run:
        mat_files = all_mat_files[:2]
        logger.info(f"Running in SAMPLE MODE: Processing first {len(mat_files)} MAT parts.")
    elif max_parts:
        mat_files = all_mat_files[:max_parts]
        logger.info(f"Running with custom limit: {len(mat_files)} MAT parts.")
    else:
        mat_files = all_mat_files
        logger.info(f"Running in FULL MODE: Processing all {len(mat_files)} MAT parts.")

    # 2. Load Phase 1 Master QC Manifest to retain master record list
    qc_manifest_path = STATISTICS_DIR / QC_MANIFEST_FILENAME
    if qc_manifest_path.exists():
        phase1_df = pd.read_csv(qc_manifest_path)
        logger.info(f"Loaded Phase 1 QC Manifest: {len(phase1_df):,} total records.")
    else:
        phase1_df = None
        logger.warning("Phase 1 QC manifest not found; will derive record IDs on the fly.")

    # 3. Process MAT parts incrementally (streaming)
    all_windows_metadata: List[Dict[str, Any]] = []
    representative_window_candidates: List[Dict[str, Any]] = []
    
    total_records_processed = 0
    records_with_windows = 0
    short_records_count = 0
    all_record_ids = []

    global_rec_counter = 0

    for part_idx, fpath in enumerate(mat_files, start=1):
        part_name = fpath.stem
        part_digits = "".join([c for c in part_name if c.isdigit()])
        part_id_str = f"part_{int(part_digits):02d}"

        logger.info(f"[{part_idx}/{len(mat_files)}] Processing {fpath.name}...")
        try:
            records_cell = load_single_mat_part(fpath)
        except Exception as e:
            logger.error(f"Failed loading {fpath.name}: {e}")
            continue

        n_records = records_cell.shape[1] if records_cell.ndim == 2 else len(records_cell)

        for rec_idx in tqdm(range(n_records), desc=f"{part_id_str}", leave=False):
            global_rec_counter += 1
            total_records_processed += 1
            rec_id_str = f"{part_id_str}_record_{global_rec_counter:06d}"
            all_record_ids.append(rec_id_str)

            try:
                rec_mat = records_cell[0, rec_idx]
                if rec_mat is None or not isinstance(rec_mat, np.ndarray) or rec_mat.ndim != 2:
                    continue

                n_channels, n_samples = rec_mat.shape
                if n_channels < 2:
                    continue

                ppg_raw = rec_mat[0, :].astype(np.float64)
                abp_raw = rec_mat[1, :].astype(np.float64)

                record_dict = {
                    "record_id": rec_id_str,
                    "part_id": part_id_str,
                    "record_index": rec_idx,
                    "global_record_index": global_rec_counter,
                    "ppg": ppg_raw,
                    "abp": abp_raw,
                    "num_samples": n_samples,
                    "sampling_frequency": SAMPLING_RATE,
                }

                # Check short records (< 1250 samples)
                if n_samples < WINDOW_SAMPLES:
                    short_records_count += 1
                    continue

                # Slice record into 10-second non-overlapping windows
                windows = slice_record_into_windows(
                    record_dict,
                    window_samples=WINDOW_SAMPLES,
                    overlap=PRIMARY_OVERLAP,
                    sampling_rate=SAMPLING_RATE,
                )

                if len(windows) > 0:
                    records_with_windows += 1

                for win in windows:
                    # 1. Window-Level PPG Quality Control
                    ppg_valid, ppg_status, ppg_reason, ppg_diag = validate_ppg_window(win["ppg"])

                    # 2. Window-Level ABP Target Extraction
                    abp_valid, abp_status, abp_reason, targets, abp_diag = extract_window_abp_targets(win["abp"])

                    # 3. Four-way Decoupled Quality Status & Modeling Eligibility
                    win_quality_status, modeling_eligible = evaluate_window_quality_status(ppg_valid, abp_valid)

                    # Build metadata dictionary (compact, no raw signal in manifest)
                    win_meta = {
                        "record_id": win["record_id"],
                        "part_id": win["part_id"],
                        "record_index": win["record_index"],
                        "window_id": win["window_id"],
                        "window_index": win["window_index"],
                        "start_sample": win["start_sample"],
                        "end_sample": win["end_sample"],
                        "start_time_seconds": win["start_time_seconds"],
                        "end_time_seconds": win["end_time_seconds"],
                        "sampling_frequency": win["sampling_frequency"],
                        "window_samples": win["window_samples"],
                        # PPG Diagnostics
                        "ppg_valid": ppg_valid,
                        "ppg_quality_status": ppg_status,
                        "ppg_rejection_reason": ppg_reason,
                        "ppg_mean": ppg_diag["ppg_mean"],
                        "ppg_std": ppg_diag["ppg_std"],
                        "ppg_ptp": ppg_diag["ppg_ptp"],
                        "ppg_min": ppg_diag["ppg_min"],
                        "ppg_max": ppg_diag["ppg_max"],
                        "ppg_clipped_fraction": ppg_diag["ppg_clipped_fraction"],
                        "ppg_pulse_count": ppg_diag["ppg_pulse_count"],
                        "estimated_hr_bpm": ppg_diag["estimated_hr_bpm"],
                        # ABP Diagnostics & Targets
                        "abp_valid": abp_valid,
                        "abp_quality_status": abp_status,
                        "abp_rejection_reason": abp_reason,
                        "valid_beat_count": abp_diag["valid_beat_count"],
                        "valid_beat_ratio": abp_diag["valid_beat_ratio"],
                        "sbp": targets["sbp"],
                        "dbp": targets["dbp"],
                        "map": targets["map"],
                        "pulse_pressure": targets["pulse_pressure"],
                        # Decoupled Quality & Eligibility
                        "window_quality_status": win_quality_status,
                        "modeling_eligible": modeling_eligible,
                    }
                    all_windows_metadata.append(win_meta)

                    # Keep a small cache of windows for representative visualization (up to 500)
                    if len(representative_window_candidates) < 500:
                        representative_window_candidates.append(win)

            except Exception as e:
                logger.warning(f"Error processing record {rec_idx} in {fpath.name}: {e}")
                continue

        # Explicit garbage collection after each MAT part
        del records_cell
        gc.collect()

    logger.info(f"Window extraction complete: {len(all_windows_metadata):,} windows extracted.")

    # 4. Construct Manifest DataFrame
    window_manifest_df = pd.DataFrame(all_windows_metadata)

    # 5. Record-Level Partitioning
    logger.info("Performing record-level train/validation/test split...")
    split_record_ids = phase1_df["record_id"].tolist() if phase1_df is not None else all_record_ids
    record_split_df = create_record_level_split(
        records=split_record_ids,
        train_ratio=SPLIT_RATIOS["train"],
        val_ratio=SPLIT_RATIOS["val"],
        test_ratio=SPLIT_RATIOS["test"],
        seed=SPLIT_RANDOM_SEED,
    )

    # Save record split CSV
    record_split_path = SPLITS_DIR / RECORD_SPLIT_FILENAME
    record_split_df.to_csv(record_split_path, index=False)
    logger.info(f"Saved record split table to {record_split_path}")

    # Annotate window manifest with partition split
    window_manifest_df = window_manifest_df.merge(
        record_split_df[["record_id", "split"]],
        on="record_id",
        how="left",
    )

    # 6. Execute 9 Mandatory Leakage & Integrity Assertions
    leakage_summary = run_leakage_and_integrity_assertions(window_manifest_df, record_split_df)
    logger.info("ALL 9 LEAKAGE & INTEGRITY ASSERTIONS PASSED SUCCESSFULLY!")

    # 7. Compute Statistics & Split Distributions
    total_win = len(window_manifest_df)
    ppg_valid_cnt = int((window_manifest_df["ppg_valid"] == True).sum())
    abp_valid_cnt = int((window_manifest_df["abp_valid"] == True).sum())
    eligible_cnt = int((window_manifest_df["modeling_eligible"] == True).sum())
    rejected_cnt = total_win - eligible_cnt

    eligible_df = window_manifest_df[window_manifest_df["modeling_eligible"] == True]

    summary_stats = {
        "total_records_processed": total_records_processed,
        "records_with_windows": records_with_windows,
        "short_records_count": short_records_count,
        "total_windows_extracted": total_win,
        "ppg_valid_count": ppg_valid_cnt,
        "ppg_valid_pct": float(ppg_valid_cnt / max(1, total_win) * 100.0),
        "abp_valid_count": abp_valid_cnt,
        "abp_valid_pct": float(abp_valid_cnt / max(1, total_win) * 100.0),
        "modeling_eligible_count": eligible_cnt,
        "modeling_eligible_pct": float(eligible_cnt / max(1, total_win) * 100.0),
        "rejected_windows_count": rejected_cnt,
        "rejected_windows_pct": float(rejected_cnt / max(1, total_win) * 100.0),
        "status_breakdown": window_manifest_df["window_quality_status"].value_counts().to_dict(),
        "target_statistics": {
            "sbp": {
                "mean": float(eligible_df["sbp"].mean()) if len(eligible_df) else np.nan,
                "std": float(eligible_df["sbp"].std()) if len(eligible_df) else np.nan,
                "min": float(eligible_df["sbp"].min()) if len(eligible_df) else np.nan,
                "p25": float(eligible_df["sbp"].quantile(0.25)) if len(eligible_df) else np.nan,
                "median": float(eligible_df["sbp"].median()) if len(eligible_df) else np.nan,
                "p75": float(eligible_df["sbp"].quantile(0.75)) if len(eligible_df) else np.nan,
                "max": float(eligible_df["sbp"].max()) if len(eligible_df) else np.nan,
            },
            "dbp": {
                "mean": float(eligible_df["dbp"].mean()) if len(eligible_df) else np.nan,
                "std": float(eligible_df["dbp"].std()) if len(eligible_df) else np.nan,
                "min": float(eligible_df["dbp"].min()) if len(eligible_df) else np.nan,
                "p25": float(eligible_df["dbp"].quantile(0.25)) if len(eligible_df) else np.nan,
                "median": float(eligible_df["dbp"].median()) if len(eligible_df) else np.nan,
                "p75": float(eligible_df["dbp"].quantile(0.75)) if len(eligible_df) else np.nan,
                "max": float(eligible_df["dbp"].max()) if len(eligible_df) else np.nan,
            },
            "map": {
                "mean": float(eligible_df["map"].mean()) if len(eligible_df) else np.nan,
                "std": float(eligible_df["map"].std()) if len(eligible_df) else np.nan,
                "min": float(eligible_df["map"].min()) if len(eligible_df) else np.nan,
                "p25": float(eligible_df["map"].quantile(0.25)) if len(eligible_df) else np.nan,
                "median": float(eligible_df["map"].median()) if len(eligible_df) else np.nan,
                "p75": float(eligible_df["map"].quantile(0.75)) if len(eligible_df) else np.nan,
                "max": float(eligible_df["map"].max()) if len(eligible_df) else np.nan,
            },
            "pp": {
                "mean": float(eligible_df["pulse_pressure"].mean()) if len(eligible_df) else np.nan,
                "std": float(eligible_df["pulse_pressure"].std()) if len(eligible_df) else np.nan,
                "min": float(eligible_df["pulse_pressure"].min()) if len(eligible_df) else np.nan,
                "median": float(eligible_df["pulse_pressure"].median()) if len(eligible_df) else np.nan,
                "max": float(eligible_df["pulse_pressure"].max()) if len(eligible_df) else np.nan,
            },
            "hr": {
                "mean": float(eligible_df["estimated_hr_bpm"].mean()) if len(eligible_df) else np.nan,
                "std": float(eligible_df["estimated_hr_bpm"].std()) if len(eligible_df) else np.nan,
                "min": float(eligible_df["estimated_hr_bpm"].min()) if len(eligible_df) else np.nan,
                "median": float(eligible_df["estimated_hr_bpm"].median()) if len(eligible_df) else np.nan,
                "max": float(eligible_df["estimated_hr_bpm"].max()) if len(eligible_df) else np.nan,
            },
        },
    }

    split_summary = compute_split_distribution_summary(window_manifest_df, record_split_df)

    # 8. Save Window Manifest CSV and Summary JSON
    manifest_save_path = WINDOWS_DIR / WINDOW_MANIFEST_FILENAME
    window_manifest_df.to_csv(manifest_save_path, index=False)
    logger.info(f"Saved window manifest table ({len(window_manifest_df):,} rows) to {manifest_save_path}")

    summary_json_path = WINDOWS_DIR / WINDOW_SUMMARY_JSON
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_statistics": summary_stats,
                "split_distribution": split_summary,
                "leakage_audit": leakage_summary,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved window summary statistics to {summary_json_path}")

    # 9. Generate Visualizations
    logger.info("Generating representative window scenario plots with manifest assertions...")
    generate_representative_window_scenarios(window_manifest_df=window_manifest_df, output_dir=FIGURES_DIR)


    logger.info("Generating dataset-level distribution figures...")
    generate_dataset_window_figures(
        window_manifest_df,
        record_split_df,
        total_raw_records=total_records_processed,
        records_with_eligible_length=records_with_windows,
        output_dir=FIGURES_DIR,
    )

    # 10. Generate Reports and Documentation
    report_path = STATISTICS_DIR / PHASE2_REPORT_FILENAME
    generate_phase2_markdown_report(summary_stats, split_summary, leakage_summary, report_path)

    readme_path = WINDOWS_DIR / "README.md"
    generate_windows_readme(readme_path)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"PHASE 2 PIPELINE COMPLETE in {elapsed:.1f} seconds ({elapsed/60.0:.2f} min).")
    logger.info("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    if args.sample_run:
        run_pipeline(sample_run=True)
    elif args.full_run:
        run_pipeline(sample_run=False)
    else:
        # Default if no flag provided: prompt or sample run
        logger.info("No run flag specified; executing sample run by default. Use --full-run for all 12 parts.")
        run_pipeline(sample_run=True)
