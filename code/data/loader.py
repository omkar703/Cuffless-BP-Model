"""
Memory-efficient Data Loader for PhysioNet MIMIC-II / Kaggle Blood Pressure Dataset.

Iterates over MATLAB cell arrays (.mat) stream-wise without loading the entire
multi-gigabyte dataset into memory simultaneously.
"""

import os
import gc
import glob
from pathlib import Path
from typing import Generator, Dict, Any, List, Optional
import numpy as np
import scipy.io as sio

from config.config import (
    DATASET_DIR,
    PPG_CHANNEL_IDX,
    ABP_CHANNEL_IDX,
    ECG_CHANNEL_IDX,
    SAMPLING_RATE,
)
from utils.logging_utils import setup_logger

logger = setup_logger("data_loader")


def get_mat_files(dataset_dir: Optional[Path] = None) -> List[Path]:
    """Discovers and returns sorted list of part_*.mat files."""
    d = dataset_dir or DATASET_DIR
    files = sorted(glob.glob(os.path.join(d, "part_*.mat")))
    # Sort numerically by part index
    def extract_part_num(p: str) -> int:
        base = os.path.basename(p)
        digits = "".join([c for c in base if c.isdigit()])
        return int(digits) if digits else 0

    files.sort(key=extract_part_num)
    return [Path(f) for f in files]


def load_single_mat_part(mat_path: Path) -> np.ndarray:
    """Loads a single .mat file and extracts the root cell array."""
    try:
        mat = sio.loadmat(str(mat_path))
        key = "p" if "p" in mat else [k for k in mat if not k.startswith("__")][0]
        records_cell = mat[key]
        return records_cell
    except Exception as e:
        logger.error(f"Failed to load MAT file {mat_path}: {e}")
        raise


def record_stream_generator(
    mat_files: Optional[List[Path]] = None,
    max_parts: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yielding individual records one by one with stable identifiers.
    
    Yields:
        dict containing:
            - record_id: Persistent unique identifier (e.g. 'part_01_record_000001')
            - part_id: Part string (e.g. 'part_01')
            - part_index: Integer index of the part file (1 to 12)
            - record_index: Local index inside the part (0 to N-1)
            - global_record_index: Global monotonic record counter
            - ppg: 1D numpy array of raw PPG samples
            - abp: 1D numpy array of raw ABP samples (mmHg)
            - num_samples: Length of signal
            - duration_seconds: num_samples / SAMPLING_RATE
            - sampling_frequency: 125 Hz
    """
    files = mat_files or get_mat_files()
    if max_parts:
        files = files[:max_parts]

    global_rec_counter = 0

    for part_idx, fpath in enumerate(files, start=1):
        part_name = fpath.stem  # e.g. 'part_1'
        # Format standardized part_id with 2-digit padding: 'part_01'
        part_digits = "".join([c for c in part_name if c.isdigit()])
        part_id_str = f"part_{int(part_digits):02d}"

        logger.info(f"Loading {fpath.name} (Part {part_idx}/{len(files)})...")
        try:
            records_cell = load_single_mat_part(fpath)
        except Exception as e:
            logger.error(f"Skipping corrupt part {fpath.name}: {e}")
            continue

        n_records = records_cell.shape[1] if records_cell.ndim == 2 else len(records_cell)

        for rec_idx in range(n_records):
            global_rec_counter += 1
            rec_id_str = f"{part_id_str}_record_{global_rec_counter:06d}"

            try:
                rec_mat = records_cell[0, rec_idx]
                # Validate 2D matrix structure
                if rec_mat is None or not isinstance(rec_mat, np.ndarray) or rec_mat.ndim != 2:
                    continue

                n_channels, n_samples = rec_mat.shape
                if n_channels < 2:
                    # Must contain at least PPG and ABP
                    continue

                # Extract PPG and ABP (convert to float64 for high-precision validation)
                ppg_raw = rec_mat[PPG_CHANNEL_IDX, :].astype(np.float64)
                abp_raw = rec_mat[ABP_CHANNEL_IDX, :].astype(np.float64)

                yield {
                    "record_id": rec_id_str,
                    "part_id": part_id_str,
                    "part_file": fpath.name,
                    "part_index": part_idx,
                    "record_index": rec_idx,
                    "global_record_index": global_rec_counter,
                    "ppg": ppg_raw,
                    "abp": abp_raw,
                    "num_samples": n_samples,
                    "duration_seconds": float(n_samples / SAMPLING_RATE),
                    "sampling_frequency": SAMPLING_RATE,
                }

            except Exception as e:
                logger.warning(f"Error reading record {rec_idx} in {fpath.name}: {e}")
                continue

        # Force memory garbage collection after each MAT part
        del records_cell
        gc.collect()
