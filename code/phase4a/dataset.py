"""
Phase 4A: PyTorch Dataset for PPG Window Loading.

Memory-conscious implementation:
- Groups windows by part_id to load each MAT file once
- Loads all eligible windows into a pre-built index (manifest rows)
- Applies preprocessing on-the-fly in __getitem__

Leakage controls enforced at __init__:
- Train / Val / Test record-ID sets must be disjoint
- Channel 2 (ECG) is never accessed
- ABP is used only as target source, never as model input
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset

# Ensure project root is on sys.path when this file is run standalone
_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from phase4a.preprocessing import preprocess_window, WINDOW_SAMPLES

# Channel indices
PPG_CHANNEL_IDX = 0
ABP_CHANNEL_IDX = 1
ECG_CHANNEL_IDX = 2  # STRICTLY FORBIDDEN as model input


def _assert_no_record_leakage(df_manifest: pd.DataFrame) -> None:
    """
    Assert that train / val / test record-ID sets are pairwise disjoint.
    Fails loudly if any overlap is detected.
    """
    splits = {s: set() for s in ["train", "val", "test"]}
    for split in splits:
        mask = df_manifest["split"] == split
        splits[split] = set(df_manifest[mask]["record_id"].unique())

    train_val = splits["train"] & splits["val"]
    train_test = splits["train"] & splits["test"]
    val_test = splits["val"] & splits["test"]

    assert len(train_val) == 0, (
        f"LEAKAGE DETECTED: {len(train_val)} record(s) appear in BOTH train and val: "
        f"{list(train_val)[:5]}"
    )
    assert len(train_test) == 0, (
        f"LEAKAGE DETECTED: {len(train_test)} record(s) appear in BOTH train and test: "
        f"{list(train_test)[:5]}"
    )
    assert len(val_test) == 0, (
        f"LEAKAGE DETECTED: {len(val_test)} record(s) appear in BOTH val and test: "
        f"{list(val_test)[:5]}"
    )


def _assert_targets_finite(sbp: np.ndarray, dbp: np.ndarray) -> None:
    """Assert that target arrays contain no NaN or Inf."""
    assert np.isfinite(sbp).all(), "NaN/Inf found in SBP targets."
    assert np.isfinite(dbp).all(), "NaN/Inf found in DBP targets."


class PPGWindowDataset(Dataset):
    """
    PyTorch Dataset for PPG windows read from MATLAB .mat files.

    Each sample is a pair (X, y) where:
        X: float32 tensor of shape (n_channels, 1250)
           n_channels=1 → PPG only (Model A)
           n_channels=3 → PPG + VPG + APG (Model B)
        y: float32 tensor of shape (2,) = [SBP, DBP]

    Loading strategy:
        The full window manifest is loaded into memory as a DataFrame.
        For each __getitem__ call, the raw PPG window is read from the
        appropriate pre-loaded numpy array (grouped by part).

    Args:
        manifest_path:  Path to window_manifest.csv
        dataset_dir:    Directory containing part_N.mat files
        split:          One of 'train', 'val', 'test'
        n_channels:     1 or 3
        use_zscore:     Apply per-window z-score normalization
        max_windows:    If set, subsample this many windows (for smoke tests)
        verify_leakage: Run record-leakage assertions at init time
    """

    def __init__(
        self,
        manifest_path: Path,
        dataset_dir: Path,
        split: str,
        n_channels: int = 3,
        use_zscore: bool = True,
        max_windows: Optional[int] = None,
        verify_leakage: bool = True,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        assert n_channels in (1, 3), f"n_channels must be 1 or 3, got {n_channels}"
        self.split = split
        self.n_channels = n_channels
        self.use_zscore = use_zscore
        self.dataset_dir = Path(dataset_dir)

        # Load full manifest for leakage check
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            gz_path = manifest_path.with_name(manifest_path.name + ".gz") if not str(manifest_path).endswith(".gz") else manifest_path
            if gz_path.exists():
                manifest_path = gz_path
            elif (manifest_path.parent / "window_manifest.csv.gz").exists():
                manifest_path = manifest_path.parent / "window_manifest.csv.gz"
            else:
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df_full = pd.read_csv(manifest_path)
        df_full = df_full[df_full["modeling_eligible"] == True].copy()

        if verify_leakage:
            _assert_no_record_leakage(df_full)

        # Filter to requested split
        df_split = df_full[df_full["split"] == split].copy()
        df_split = df_split.reset_index(drop=True)

        if max_windows is not None:
            df_split = df_split.iloc[:max_windows].copy()

        # Validate targets
        _assert_targets_finite(df_split["sbp"].values, df_split["dbp"].values)

        self.manifest = df_split

        # Build the raw PPG array: pre-load all windows into numpy
        # This is memory-efficient because we load one MAT part at a time
        print(f"[PPGWindowDataset] Building index for split='{split}', "
              f"n_windows={len(df_split)}, n_channels={n_channels}...")
        self._ppg_data, self._sbp, self._dbp = self._load_all_windows(df_split)
        print(f"[PPGWindowDataset] Loaded {len(self._ppg_data)} windows. "
              f"PPG shape: {self._ppg_data.shape}")

    def _load_all_windows(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all raw PPG windows from MAT files, grouped by part_id.
        Returns:
            ppg_array: (N, 1250) float64
            sbp_array: (N,) float32
            dbp_array: (N,) float32
        """
        n = len(df)
        ppg_array = np.empty((n, WINDOW_SAMPLES), dtype=np.float32)
        sbp_array = df["sbp"].values.astype(np.float32)
        dbp_array = df["dbp"].values.astype(np.float32)

        # Process one part at a time
        for part_id, group in df.groupby("part_id", sort=True):
            part_num = int("".join([c for c in part_id if c.isdigit()]))
            mat_path = self.dataset_dir / f"part_{part_num}.mat"
            if not mat_path.exists():
                mat_path = self.dataset_dir / f"part_{part_num:02d}.mat"
            if not mat_path.exists():
                raise FileNotFoundError(f"MAT file not found: {mat_path}")

            print(f"  Loading {mat_path.name} ({len(group)} windows)...")
            mat = sio.loadmat(str(mat_path))
            key = "p" if "p" in mat else [k for k in mat if not k.startswith("__")][0]
            records_cell = mat[key]

            for rec_idx, rec_group in group.groupby("record_index", sort=True):
                rec_mat = records_cell[0, rec_idx]
                if rec_mat is None or not isinstance(rec_mat, np.ndarray) or rec_mat.ndim != 2:
                    raise RuntimeError(
                        f"Invalid record {rec_idx} in {part_id}. "
                        "Expected 2D ndarray."
                    )

                # STRICT: only access channel 0 (PPG)
                assert rec_mat.shape[0] >= 1, "Record has no channels."
                ppg_full = rec_mat[PPG_CHANNEL_IDX, :]

                # STRICT: assert we never touch channel 2 (ECG)
                # (We simply never index it; this comment documents the invariant)
                # assert PPG_CHANNEL_IDX != ECG_CHANNEL_IDX  # always true: 0 != 2

                starts = rec_group["start_sample"].values.astype(int)
                ends = rec_group["end_sample"].values.astype(int)
                row_indices = rec_group.index.values

                for df_row_idx, start, end in zip(row_indices, starts, ends):
                    win = ppg_full[start:end].astype(np.float32)

                    if len(win) != WINDOW_SAMPLES:
                        raise RuntimeError(
                            f"Window at index {df_row_idx} has {len(win)} samples, "
                            f"expected {WINDOW_SAMPLES}."
                        )
                    if not np.isfinite(win).all():
                        raise RuntimeError(
                            f"NaN/Inf in window at index {df_row_idx}."
                        )

                    ppg_array[df_row_idx] = win

            del records_cell, mat
            gc.collect()

        return ppg_array, sbp_array, dbp_array

    def __len__(self) -> int:
        return len(self._sbp)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: torch.FloatTensor of shape (n_channels, 1250)
            y: torch.FloatTensor of shape (2,) = [SBP, DBP]
        """
        raw_ppg = self._ppg_data[idx].astype(np.float64)

        # Preprocess: filter + derivatives + zscore
        x = preprocess_window(raw_ppg, use_zscore=self.use_zscore, n_channels=self.n_channels)
        # x shape: (n_channels, 1250), float32

        # Final input validation
        assert np.isfinite(x).all(), f"NaN/Inf in preprocessed window at index {idx}"

        y = np.array([self._sbp[idx], self._dbp[idx]], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def record_ids(self) -> np.ndarray:
        """Return array of record IDs for this split."""
        return self.manifest["record_id"].values
