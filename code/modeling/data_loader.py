"""
Data Loader for Phase 3A PPG-Only Baseline Modeling.

Extracts Channel 0 (PPG) window slices from raw MATLAB .mat files according
to the frozen Phase 2 window manifest. Channel 2 (ECG) is strictly prohibited.
ABP is never exposed to feature extraction and is used solely for ground truth targets.
"""

import os
import gc
from pathlib import Path
from typing import Generator, Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.io as sio

from config.config import (
    DATASET_DIR,
    WINDOWS_DIR,
    WINDOW_MANIFEST_FILENAME,
    PPG_CHANNEL_IDX,
    ABP_CHANNEL_IDX,
    ECG_CHANNEL_IDX,
    SAMPLING_RATE,
    WINDOW_SAMPLES,
)
from utils.logging_utils import setup_logger

logger = setup_logger("modeling_data_loader")

# Constant assertion: Channel 0 is PPG
assert PPG_CHANNEL_IDX == 0, f"PPG_CHANNEL_IDX must be 0, found {PPG_CHANNEL_IDX}"


def load_eligible_manifest(
    manifest_path: Optional[Path] = None,
    splits: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Loads window_manifest.csv and filters to modeling-eligible windows.
    Optionally filters by split list (e.g. ['train'], ['val'], ['test']).
    """
    p = manifest_path or (WINDOWS_DIR / WINDOW_MANIFEST_FILENAME)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    df = pd.read_csv(p)
    logger.info(f"Loaded raw manifest with {len(df)} rows.")

    # Filter to modeling eligible
    df_eligible = df[df["modeling_eligible"] == True].copy()
    logger.info(f"Filtered to {len(df_eligible)} modeling-eligible windows.")

    if splits is not None:
        df_eligible = df_eligible[df_eligible["split"].isin(splits)].copy()
        logger.info(f"Filtered by splits {splits}: {len(df_eligible)} windows remain.")

    return df_eligible


def stream_window_ppg(
    df_manifest: pd.DataFrame,
    dataset_dir: Optional[Path] = None,
    max_windows: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator streaming individual PPG windows (Channel 0 only) from raw MAT part files.
    
    Yields dict:
        - record_id: str
        - window_id: str
        - window_index: int
        - split: str
        - ppg: np.ndarray of shape (1250,)
        - sbp: float
        - dbp: float
        - map: float
        - ppg_quality_status: str
        - window_quality_status: str
    """
    base_dir = dataset_dir or DATASET_DIR
    yielded_count = 0

    # Group manifest by part_id for efficient disk reading (load each MAT part once)
    grouped = df_manifest.groupby("part_id", sort=True)

    for part_id, group in grouped:
        part_num = int("".join([c for c in part_id if c.isdigit()]))
        mat_path = base_dir / f"part_{part_num}.mat"
        if not mat_path.exists():
            # Try 2-digit naming if present
            mat_path = base_dir / f"part_{part_num:02d}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Part MAT file not found: {mat_path}")

        logger.info(f"Streaming windows from {mat_path.name} ({len(group)} windows)...")
        mat = sio.loadmat(str(mat_path))
        key = "p" if "p" in mat else [k for k in mat if not k.startswith("__")][0]
        records_cell = mat[key]

        # Group by record_index within this part
        rec_grouped = group.groupby("record_index", sort=True)

        for rec_idx, win_rows in rec_grouped:
            rec_mat = records_cell[0, rec_idx]
            if rec_mat is None or not isinstance(rec_mat, np.ndarray) or rec_mat.ndim != 2:
                logger.warning(f"Record {rec_idx} in {part_id} is invalid; skipping.")
                continue

            # Verify and extract Channel 0 strictly
            for _, row in win_rows.iterrows():
                start = int(row["start_sample"])
                end = int(row["end_sample"])

                # Strictly extract PPG (Channel 0)
                ppg_window = rec_mat[PPG_CHANNEL_IDX, start:end].astype(np.float64)

                # Assertions
                if len(ppg_window) != WINDOW_SAMPLES:
                    logger.warning(
                        f"Window {row['window_id']} has length {len(ppg_window)} != {WINDOW_SAMPLES}; skipping."
                    )
                    continue

                yield {
                    "record_id": row["record_id"],
                    "window_id": row["window_id"],
                    "window_index": int(row["window_index"]),
                    "split": row["split"],
                    "ppg": ppg_window,
                    "sbp": float(row["sbp"]),
                    "dbp": float(row["dbp"]),
                    "map": float(row["map"]),
                    "ppg_quality_status": row.get("ppg_quality_status", "PASS"),
                    "window_quality_status": row.get("window_quality_status", "PPG_VALID_ABP_VALID"),
                    "ppg_clipped_fraction": float(row.get("ppg_clipped_fraction", 0.0)),
                    "estimated_hr_bpm": float(row.get("estimated_hr_bpm", 0.0)),
                }

                yielded_count += 1
                if max_windows and yielded_count >= max_windows:
                    del records_cell, mat
                    gc.collect()
                    return

        del records_cell, mat
        gc.collect()
