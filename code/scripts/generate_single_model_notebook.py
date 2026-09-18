"""
Script to generate code/notebooks/04A_neural_ppg_vpg_apg_single_model.ipynb
Clean single-model Phase 4A notebook:
PPG + VPG + APG -> 1D CNN -> SBP + DBP
"""

from pathlib import Path
import nbformat as nbf

def generate_single_model_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3 (.venv)"
        }
    }

    cells = []

    # =========================================================================
    # Cell 1: Markdown Header & Research Objective
    # =========================================================================
    c1 = nbf.v4.new_markdown_cell("""# PHASE 4A — Neural PPG Baseline (Single-Model Experiment)
## 10-second PPG + VPG + APG 1D CNN for Calibration-Free Cuffless Blood Pressure Estimation

**Research Question:**
> *Can a compact 3-channel neural representation consisting of PPG, VPG, and APG estimate SBP and DBP from a single 10-second PPG window under the existing calibration-free, record-level leakage-controlled protocol?*

---

### Key Experimental Invariants:
1. **Single Neural Model**:
   - Model: `model_ppg_vpg_apg` (3 input channels: PPG + VPG + APG).
   - This experiment does NOT perform a dual-model neural ablation. The neural representation is evaluated directly against the frozen classical baseline.
2. **Frozen External Benchmark (Phase 3A Classical HistGB)**:
   - SBP MAE: **13.93 mmHg**
   - DBP MAE: **7.05 mmHg**
   - Combined MAE: **10.49 mmHg**
3. **Sensor Constraints**:
   - Strictly 10-second PPG waveforms ($F_s = 125$ Hz, 1,250 samples).
   - **ECG (channel 2) is strictly forbidden** (never loaded, never indexed, no PTT, no ECG features).
   - **ABP (channel 1) is strictly ground truth target**, never model input.
4. **Calibration-Free**: Zero subject-specific baseline BP tuning or recalibration.
5. **Leakage Control**: Disjoint record-level partition (Train: 8,400 records, Val: 1,800 records, Test: 1,800 records).
6. **Isolated Output Directory**: `code/outputs/phase4a_single_model/`
""")
    cells.append(c1)

    # =========================================================================
    # Cell 2: Environment & Hardware Setup
    # =========================================================================
    c2 = nbf.v4.new_code_cell("""# 1. Environment & Hardware Verification
import os
import sys
import time
import gc
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample_poly

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Hardware Detection
print("=" * 70)
print("SYSTEM & HARDWARE SPECIFICATIONS")
print("=" * 70)
print(f"Python Version:  {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM:      {total_vram:.2f} GB")
    device = torch.device("cuda")
else:
    print("Using CPU (CUDA not available)")
    device = torch.device("cpu")
print("=" * 70)

# Deterministic Seeding
SEED = 42
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"Deterministic random seed locked to {SEED}.")
""")
    cells.append(c2)

    # =========================================================================
    # Cell 3: Sampling Rate Strategy & Deployment Utility
    # =========================================================================
    c3 = nbf.v4.new_code_cell("""# 2. Sampling Rate Decision & Hardware Deployment Strategy
#
# RESEARCH DECISION:
# The research model and dataset remain strictly at Fs = 125 Hz (1,250 samples / 10s).
# Training data is NOT converted or downsampled to 100 Hz.
#
# FUTURE HARDWARE DEPLOYMENT STRATEGY:
# The target hardware sensor (MAX30102) typically samples at 100 Hz.
# For embedded deployment at inference time:
#   1. Acquire 10s PPG window at 100 Hz (1,000 samples)
#   2. Apply polyphase anti-aliasing resampling: 100 Hz -> 125 Hz (1,250 samples)
#      Ratio: 125 / 100 = 5 / 4
#   3. Apply identical 125 Hz bandpass (0.5 - 8.0 Hz) and derivative processing
#   4. Pass into this trained 125 Hz neural model
#
# Below is the deployment resampling utility for documentation.
# STRICT INVARIANT: This utility is NOT called during Phase 4A training.

def resample_100_to_125(ppg_100hz: np.ndarray) -> np.ndarray:
    \"\"\"
    Polyphase anti-aliasing resampling from 100 Hz to 125 Hz.
    125 / 100 = 5 / 4 (up=5, down=4).
    Anti-aliasing filtering is performed internally by resample_poly.
    \"\"\"
    return resample_poly(ppg_100hz, up=5, down=4)

print("Sampling rate protocol: Fs = 125 Hz (1,250 samples / 10s window).")
print("Hardware deployment polyphase resampling function compiled (for future deployment only).")
""")
    cells.append(c3)

    # =========================================================================
    # Cell 4: Workspace Directories & Local Data Discovery
    # =========================================================================
    c4 = nbf.v4.new_code_cell("""# 3. Clean Output Directory Layout & Local Path Discovery
cwd = Path.cwd().resolve()
if (cwd / "BloodPressureDataset").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "BloodPressureDataset").exists():
    PROJECT_ROOT = cwd.parent
elif (cwd.parent.parent / "BloodPressureDataset").exists():
    PROJECT_ROOT = cwd.parent.parent
else:
    PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone").resolve()

CODE_DIR = PROJECT_ROOT / "code"
DATASET_DIR = PROJECT_ROOT / "BloodPressureDataset"

# Dedicated clean output directory for single-model Phase 4A
OUTPUT_DIR = CODE_DIR / "outputs" / "phase4a_single_model"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

for d in [CKPT_DIR, METRICS_DIR, PRED_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Dataset Dir:  {DATASET_DIR}")
print(f"Output Dir:   {OUTPUT_DIR}")

# Verify presence of all 12 raw MAT files
mat_files = sorted(list(DATASET_DIR.glob("part_*.mat")))
print(f"\\nFound {len(mat_files)} local MAT file(s) in {DATASET_DIR.name}:")
for f in mat_files[:4]:
    print(f"  - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")
if len(mat_files) > 4:
    print(f"  ... and {len(mat_files) - 4} more.")
assert len(mat_files) >= 12, f"Expected 12 MAT files, found {len(mat_files)}"
""")
    cells.append(c4)

    # =========================================================================
    # Cell 5: Manifest Loading & Zero-Leakage Record Partitioning
    # =========================================================================
    c5 = nbf.v4.new_code_cell("""# 4. Load Window Manifest & Enforce Zero-Leakage Checks
# Manifest contains physiological windows extracted during Phase 2.
# Loads window_manifest.csv.gz or window_manifest.csv directly.

manifest_candidates = [
    CODE_DIR / "outputs" / "windows" / "window_manifest.csv.gz",
    CODE_DIR / "outputs" / "windows" / "window_manifest.csv",
    PROJECT_ROOT / "window_manifest.csv.gz",
]
manifest_path = None
for p in manifest_candidates:
    if p.exists():
        manifest_path = p
        break

assert manifest_path is not None, "Window manifest file not found!"
print(f"Loading manifest from: {manifest_path}...")
t0 = time.time()
df_full = pd.read_csv(manifest_path)
df_eligible = df_full[df_full["modeling_eligible"] == True].copy().reset_index(drop=True)
print(f"Loaded {len(df_eligible):,} eligible windows in {time.time() - t0:.2f}s")

# -------------------------------------------------------------------------
# STRICT LEAKAGE CONTROL ASSERTIONS
# -------------------------------------------------------------------------
train_recs = set(df_eligible[df_eligible["split"] == "train"]["record_id"].unique())
val_recs = set(df_eligible[df_eligible["split"] == "val"]["record_id"].unique())
test_recs = set(df_eligible[df_eligible["split"] == "test"]["record_id"].unique())

assert len(train_recs & val_recs) == 0, "CRITICAL ERROR: Leakage between Train and Val sets!"
assert len(train_recs & test_recs) == 0, "CRITICAL ERROR: Leakage between Train and Test sets!"
assert len(val_recs & test_recs) == 0, "CRITICAL ERROR: Leakage between Val and Test sets!"

# Target sanity checks
assert np.isfinite(df_eligible["sbp"].values).all(), "Non-finite SBP target found!"
assert np.isfinite(df_eligible["dbp"].values).all(), "Non-finite DBP target found!"

n_train = len(df_eligible[df_eligible['split'] == 'train'])
n_val = len(df_eligible[df_eligible['split'] == 'val'])
n_test = len(df_eligible[df_eligible['split'] == 'test'])

print("=" * 60)
print("ZERO-LEAKAGE RECORD PARTITION VERIFIED:")
print(f"  Train: {len(train_recs):>5} records | {n_train:>7} eligible windows")
print(f"  Val:   {len(val_recs):>5} records | {n_val:>7} eligible windows")
print(f"  Test:  {len(test_recs):>5} records | {n_test:>7} eligible windows")
print(f"  Total: {len(train_recs)+len(val_recs)+len(test_recs):>5} records | {len(df_eligible):>7} eligible windows")
print("=" * 60)
""")
    cells.append(c5)

    # =========================================================================
    # Cell 6: Preprocessing Pipeline (PPG + VPG + APG)
    # =========================================================================
    c6 = nbf.v4.new_code_cell("""# 5. Preprocessing Pipeline: PPG, VPG, and APG
#
# Preprocessing Protocol:
#   1. Zero-phase Butterworth bandpass filter (0.5 - 8.0 Hz, 3rd order, filtfilt)
#   2. VPG = 1st derivative (np.gradient with dt = 1/125s)
#   3. APG = 2nd derivative (np.gradient of VPG with dt = 1/125s)
#   4. Per-channel per-window z-score normalization: z = (x - mean) / (std + 1e-8)
#
# TERMINOLOGY NOTE:
# VPG and APG represent morphological derivative representations of the PPG waveform.
# They are not interpreted as literal physical blood-flow velocity or acceleration.

FS: float = 125.0
LOWCUT: float = 0.5
HIGHCUT: float = 8.0
FILTER_ORDER: int = 3
WINDOW_SAMPLES: int = 1250
ZSCORE_EPS: float = 1e-8

def bandpass_filter(ppg: np.ndarray, fs: float = FS, lowcut: float = LOWCUT, highcut: float = HIGHCUT, order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, ppg)

def compute_vpg(ppg_filtered: np.ndarray, dt: float = 1.0 / FS) -> np.ndarray:
    return np.gradient(ppg_filtered, dt)

def compute_apg(vpg: np.ndarray, dt: float = 1.0 / FS) -> np.ndarray:
    return np.gradient(vpg, dt)

def per_window_zscore(channel: np.ndarray, eps: float = ZSCORE_EPS) -> np.ndarray:
    return (channel - np.mean(channel)) / (np.std(channel) + eps)

def preprocess_window(raw_ppg: np.ndarray, use_zscore: bool = True) -> np.ndarray:
    \"\"\"
    Takes raw 10-second PPG waveform (1250 samples).
    Returns 3-channel preprocessed tensor of shape [3, 1250], float32:
      Channel 0: Bandpass filtered PPG
      Channel 1: VPG (1st derivative)
      Channel 2: APG (2nd derivative)
    \"\"\"
    raw_ppg = np.asarray(raw_ppg, dtype=np.float64)
    if len(raw_ppg) != WINDOW_SAMPLES:
        raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {len(raw_ppg)}")
    
    dt = 1.0 / FS
    ppg_filt = bandpass_filter(raw_ppg, fs=FS)
    vpg = compute_vpg(ppg_filt, dt=dt)
    apg = compute_apg(vpg, dt=dt)
    
    if use_zscore:
        ppg_filt = per_window_zscore(ppg_filt)
        vpg = per_window_zscore(vpg)
        apg = per_window_zscore(apg)
        
    return np.stack([ppg_filt, vpg, apg], axis=0).astype(np.float32)

print("Preprocessing pipeline compiled. Input shape: [3, 1250].")
""")
    cells.append(c6)

    # =========================================================================
    # Cell 7: PyTorch Dataset (Strict Sensor Isolation)
    # =========================================================================
    c7 = nbf.v4.new_code_cell("""# 6. PyTorch Dataset with Strict Sensor Isolation
#
# SENSOR INTEGRITY RULES:
#   - Channel 0 = PPG (Model input source)
#   - Channel 1 = ABP (Ground-truth SBP/DBP targets only, NEVER input)
#   - Channel 2 = ECG (STRICTLY FORBIDDEN: never loaded, never indexed)

PPG_CHANNEL_IDX = 0
ABP_CHANNEL_IDX = 1
ECG_CHANNEL_IDX = 2

assert PPG_CHANNEL_IDX == 0
assert ECG_CHANNEL_IDX == 2
assert PPG_CHANNEL_IDX != ECG_CHANNEL_IDX, "Invariant violation: PPG and ECG channel indices match!"

class PPGWindowDataset(Dataset):
    def __init__(
        self,
        df_manifest: pd.DataFrame,
        dataset_dir: Path,
        split: str,
        use_zscore: bool = True,
        max_windows: Optional[int] = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        self.use_zscore = use_zscore
        self.dataset_dir = Path(dataset_dir)
        
        df_split = df_manifest[df_manifest["split"] == split].copy().reset_index(drop=True)
        if max_windows is not None:
            df_split = df_split.iloc[:max_windows].copy().reset_index(drop=True)
            
        self.manifest = df_split
        print(f"[{split.upper()}] Loading {len(df_split):,} raw PPG windows from MAT parts...")
        self._ppg_data, self._sbp, self._dbp = self._load_windows(df_split)
        print(f"[{split.upper()}] Memory buffer ready: {self._ppg_data.shape}, dtype={self._ppg_data.dtype}")

    def _load_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        ppg_array = np.empty((n, WINDOW_SAMPLES), dtype=np.float32)
        sbp_array = df["sbp"].values.astype(np.float32)
        dbp_array = df["dbp"].values.astype(np.float32)
        
        # Load one MAT part at a time to prevent RAM thrashing
        for part_id, group in df.groupby("part_id", sort=True):
            part_num = int("".join([c for c in part_id if c.isdigit()]))
            mat_path = self.dataset_dir / f"part_{part_num}.mat"
            if not mat_path.exists():
                mat_path = self.dataset_dir / f"part_{part_num:02d}.mat"
            if not mat_path.exists():
                raise FileNotFoundError(f"MAT file not found: {mat_path}")
                
            mat = sio.loadmat(str(mat_path))
            key = "p" if "p" in mat else [k for k in mat if not k.startswith("__")][0]
            records_cell = mat[key]
            
            for rec_idx, rec_group in group.groupby("record_index", sort=True):
                rec_mat = records_cell[0, rec_idx]
                
                # STRICT SENSOR ASSERTION: Only extract Channel 0 (PPG). Channel 2 (ECG) is never accessed.
                assert rec_mat.ndim == 2, f"Expected 2D record matrix, got {rec_mat.ndim}"
                ppg_full = rec_mat[PPG_CHANNEL_IDX, :]
                
                starts = rec_group["start_sample"].values.astype(int)
                ends = rec_group["end_sample"].values.astype(int)
                indices = rec_group.index.values
                
                for idx, s, e in zip(indices, starts, ends):
                    ppg_array[idx] = ppg_full[s:e].astype(np.float32)
                    
            del records_cell, mat
            gc.collect()
            
        return ppg_array, sbp_array, dbp_array

    def __len__(self) -> int:
        return len(self._sbp)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_ppg = self._ppg_data[idx]
        x = preprocess_window(raw_ppg, use_zscore=self.use_zscore)  # [3, 1250]
        y = np.array([self._sbp[idx], self._dbp[idx]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

print("PPGWindowDataset compiled with strict PPG-only channel loading.")
""")
    cells.append(c7)

    # =========================================================================
    # Cell 8: Single-Model 1D CNN Architecture (PPGCNNBaseline)
    # =========================================================================
    c8 = nbf.v4.new_code_cell("""# 7. Single 1D CNN Architecture (PPGCNNBaseline)
#
# Fixed 3-channel input: PPG + VPG + APG -> [B, 3, 1250]
# Feature Extractor: 4 ConvBlocks (Conv1D + BatchNorm + ReLU + Pooling)
# Head: AdaptiveAvgPool1D(1) -> FC(128 -> 64) -> Dropout(0.2) -> Dual Linear Heads (SBP, DBP)

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int = 2, use_adaptive_pool: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same", bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if use_adaptive_pool:
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))

class PPGCNNBaseline(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.n_channels = 3  # STRICTLY 3 CHANNELS: PPG + VPG + APG
        
        # 4 Convolutional Feature Extraction Blocks
        self.block1 = ConvBlock(3, 32, kernel_size=7, pool_size=2)
        self.block2 = ConvBlock(32, 64, kernel_size=7, pool_size=2)
        self.block3 = ConvBlock(64, 128, kernel_size=5, pool_size=2)
        self.block4 = ConvBlock(128, 128, kernel_size=5, use_adaptive_pool=True)
        
        # Dense Dual Regression Heads
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.sbp_head = nn.Linear(64, 1)
        self.dbp_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.block4(self.block3(self.block2(self.block1(x))))
        feat = self.fc(self.flatten(feat))
        sbp = self.sbp_head(feat)
        dbp = self.dbp_head(feat)
        return torch.cat([sbp, dbp], dim=1)  # [B, 2]

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# Instantiate and verify parameter counts
model_demo = PPGCNNBaseline()
total_p, train_p = model_demo.count_parameters()
print("=" * 60)
print(f"MODEL: PPGCNNBaseline (PPG + VPG + APG, 3 Channels)")
print(f"  Total parameters:     {total_p:,}")
print(f"  Trainable parameters: {train_p:,}")
print("=" * 60)
assert total_p == 146978, f"Expected 146,978 parameters, got {total_p}"
""")
    cells.append(c8)

    # =========================================================================
    # Cell 9: Smoke Test Execution
    # =========================================================================
    c9 = nbf.v4.new_code_cell("""# 8. Smoke Test Execution
# Verifies forward pass shape, backward pass, gradient finiteness, and loss reduction on 128 windows.

def run_smoke_test(device, manifest_df, dataset_dir):
    print("=" * 60)
    print("RUNNING PIPELINE SMOKE TEST")
    print("=" * 60)
    set_seed(42)
    
    ds_smoke = PPGWindowDataset(
        manifest_df, dataset_dir, split="train",
        use_zscore=True, max_windows=128
    )
    loader_smoke = DataLoader(ds_smoke, batch_size=32, shuffle=True)
    
    model = PPGCNNBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=5.0)
    
    # 1. Forward Pass Check
    model.train()
    xb, yb = next(iter(loader_smoke))
    xb, yb = xb.to(device), yb.to(device)
    out = model(xb)
    assert out.shape == (xb.shape[0], 2), f"Expected shape ({xb.shape[0]}, 2), got {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf detected in model output!"
    print(f"1. Forward Pass: Output shape {out.shape} [OK]")
    
    # 2. Backward Pass Check
    optimizer.zero_grad()
    loss = criterion(out[:, 0], yb[:, 0]) + criterion(out[:, 1], yb[:, 1])
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"
    print(f"2. Backward Pass: Gradients verified finite [OK]")
    
    # 3. 2 Mini-Epochs Check
    losses = []
    for epoch in range(2):
        ep_loss = 0.0
        for x_step, y_step in loader_smoke:
            x_step, y_step = x_step.to(device), y_step.to(device)
            optimizer.zero_grad()
            pred = model(x_step)
            l = criterion(pred[:, 0], y_step[:, 0]) + criterion(pred[:, 1], y_step[:, 1])
            l.backward()
            optimizer.step()
            ep_loss += l.item()
        avg_l = ep_loss / len(loader_smoke)
        losses.append(avg_l)
        print(f"   Mini-epoch {epoch+1}/2 Loss: {avg_l:.4f}")
        
    assert losses[1] <= losses[0], "Warning: Loss did not trend downward in mini-epochs"
    print(f"3. Mini-epochs: Loss decreased from {losses[0]:.4f} to {losses[1]:.4f} [OK]")
    print("=" * 60)
    print("SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 60)

run_smoke_test(device, df_eligible, DATASET_DIR)
""")
    cells.append(c9)

    # =========================================================================
    # Cell 10: Training Configuration, Loss, & Metrics
    # =========================================================================
    c10 = nbf.v4.new_code_cell("""# 9. Training Engine & Evaluation Utilities
HUBER_DELTA = 5.0
BATCH_SIZE = 128  # Memory-conscious for 4GB VRAM GTX 1650 Ti (fallback to 64 if needed)
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Frozen Classical Benchmark (Phase 3A HistGB)
PHASE3A_SBP_MAE = 13.93
PHASE3A_DBP_MAE = 7.05
PHASE3A_COMB_MAE = 10.49

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = len(y_true)
    var_true = float(np.var(y_true))
    r2 = float(1.0 - np.sum(errors ** 2) / (n * var_true)) if var_true > 1e-9 else 0.0
    
    pct_5 = float(np.mean(abs_errors <= 5.0) * 100)
    pct_10 = float(np.mean(abs_errors <= 10.0) * 100)
    pct_15 = float(np.mean(abs_errors <= 15.0) * 100)
    
    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": r2,
        "bias": float(np.mean(errors)),
        "error_sd": float(np.std(errors)),
        "pct_within_5": pct_5,
        "pct_within_10": pct_10,
        "pct_within_15": pct_15,
        "n_samples": n,
    }

def compute_bp_range_stratification(y_true: np.ndarray, y_pred: np.ndarray, target: str = "SBP") -> pd.DataFrame:
    abs_err = np.abs(y_pred - y_true)
    err = y_pred - y_true
    if target.upper() == "SBP":
        bins = [-np.inf, 90, 120, 140, 160, np.inf]
        labels = ["<90", "90-119", "120-139", "140-159", ">=160"]
    else:
        bins = [-np.inf, 60, 80, 90, 100, np.inf]
        labels = ["<60", "60-79", "80-89", "90-99", ">=100"]
    
    cats = pd.cut(y_true, bins=bins, labels=labels, right=False)
    rows = []
    for lbl in labels:
        m = (cats == lbl)
        cnt = int(np.sum(m))
        rows.append({
            "target": target, "range": lbl, "sample_count": cnt,
            "mae": float(np.mean(abs_err[m])) if cnt > 0 else np.nan,
            "rmse": float(np.sqrt(np.mean(err[m]**2))) if cnt > 0 else np.nan,
            "bias": float(np.mean(err[m])) if cnt > 0 else np.nan,
            "error_sd": float(np.std(err[m])) if cnt > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def compute_record_level_metrics(df_manifest: pd.DataFrame, sbp_true: np.ndarray, sbp_pred: np.ndarray, dbp_true: np.ndarray, dbp_pred: np.ndarray) -> pd.DataFrame:
    \"\"\"
    Record-independent evaluation: aggregates window predictions by record_id.
    \"\"\"
    df_eval = pd.DataFrame({
        "record_id": df_manifest["record_id"].values,
        "sbp_true": sbp_true,
        "sbp_pred": sbp_pred,
        "dbp_true": dbp_true,
        "dbp_pred": dbp_pred,
    })
    
    rec_rows = []
    for rec_id, group in df_eval.groupby("record_id"):
        rec_sbp_mae = np.mean(np.abs(group["sbp_pred"] - group["sbp_true"]))
        rec_dbp_mae = np.mean(np.abs(group["dbp_pred"] - group["dbp_true"]))
        rec_rows.append({
            "record_id": rec_id,
            "window_count": len(group),
            "sbp_mae": float(rec_sbp_mae),
            "dbp_mae": float(rec_dbp_mae),
            "comb_mae": float((rec_sbp_mae + rec_dbp_mae) / 2.0)
        })
    return pd.DataFrame(rec_rows)

def train_single_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
):
    set_seed(42)
    print("=" * 70)
    print("TRAINING: Single-Model PPG + VPG + APG 1D CNN")
    print("=" * 70)
    
    model = PPGCNNBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    
    best_val_comb = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses, val_losses = [], []
    history = []
    ckpt_path = CKPT_DIR / "best_model_ppg_vpg_apg.pt"
    
    log_file = LOG_DIR / "training_log.txt"
    with open(log_file, "w") as f:
        f.write("=== Training Log: Phase 4A Single-Model PPG+VPG+APG 1D CNN ===\\n")
    
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        # 1. Training Pass
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred[:, 0], yb[:, 0]) + criterion(pred[:, 1], yb[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # 2. Validation Pass
        model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                l = criterion(pred[:, 0], yb[:, 0]) + criterion(pred[:, 1], yb[:, 1])
                val_loss += l.item()
                val_true.append(yb.cpu().numpy())
                val_pred.append(pred.cpu().numpy())
                
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        
        val_true = np.vstack(val_true)
        val_pred = np.vstack(val_pred)
        sbp_mae = float(np.mean(np.abs(val_pred[:, 0] - val_true[:, 0])))
        dbp_mae = float(np.mean(np.abs(val_pred[:, 1] - val_true[:, 1])))
        comb_mae = (sbp_mae + dbp_mae) / 2.0
        
        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(comb_mae)
        
        is_best = comb_mae < best_val_comb
        if is_best:
            best_val_comb = comb_mae
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_comb_mae": comb_mae,
                "val_sbp_mae": sbp_mae,
                "val_dbp_mae": dbp_mae,
                "val_true": val_true,
                "val_pred": val_pred,
                "preprocessing_config": {
                    "fs": FS, "lowcut": LOWCUT, "highcut": HIGHCUT, "order": FILTER_ORDER,
                    "window_samples": WINDOW_SAMPLES, "zscore": True
                },
                "sampling_rate": FS,
                "channel_definitions": ["PPG", "VPG", "APG"],
                "random_seed": SEED,
            }, ckpt_path)
            mark = "BEST *"
        else:
            no_improve += 1
            mark = f"NoImprove={no_improve}"
            
        msg = (f"Epoch {epoch:02d}/{max_epochs:02d} | TrainLoss={avg_train:.3f} | ValLoss={avg_val:.3f} | "
               f"SBP={sbp_mae:.2f} DBP={dbp_mae:.2f} Comb={comb_mae:.2f} mmHg | LR={curr_lr:.1e} | {mark}")
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\\n")
        
        history.append({
            "epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
            "val_sbp_mae": sbp_mae, "val_dbp_mae": dbp_mae, "val_comb_mae": comb_mae, "lr": curr_lr
        })
        
        if no_improve >= patience:
            print(f"\\nEarly stopping triggered after {patience} epochs without improvement.")
            break
            
    elapsed = (time.time() - start_time) / 60.0
    print(f"\\nTraining finished in {elapsed:.1f} min. Best model at epoch {best_epoch} with Val Comb MAE={best_val_comb:.2f} mmHg.")
    
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model, best_ckpt, history, train_losses, val_losses

train_model = train_single_model  # alias for backwards compatibility
print("Training engine & evaluation metrics compiled.")
""")
    cells.append(c10)

    # =========================================================================
    # Cell 11: Build DataLoaders
    # =========================================================================
    c11 = nbf.v4.new_code_cell("""# 10. Instantiate Split Datasets & DataLoaders
# Uses num_workers=0 to conserve RAM (fits comfortably within 8GB system RAM)
# Pin memory is enabled for CUDA transfers.

print("Building Dataset splits (3 channels: PPG + VPG + APG)...")
train_ds = PPGWindowDataset(df_eligible, DATASET_DIR, split="train")
val_ds   = PPGWindowDataset(df_eligible, DATASET_DIR, split="val")
test_ds  = PPGWindowDataset(df_eligible, DATASET_DIR, split="test")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

print("\\nDataLoaders ready:")
print(f"  Train: {len(train_ds):,} windows ({len(train_loader)} batches @ batch_size={BATCH_SIZE})")
print(f"  Val:   {len(val_ds):,} windows ({len(val_loader)} batches @ batch_size={BATCH_SIZE*2})")
print(f"  Test:  {len(test_ds):,} windows ({len(test_loader)} batches @ batch_size={BATCH_SIZE*2})")
""")
    cells.append(c11)

    # =========================================================================
    # Cell 12: Full Training Execution & Safe Checkpoint Recovery
    # =========================================================================
    c12 = nbf.v4.new_code_cell("""# 11. Full Training Run & Checkpoint Auto-Recovery
# If training was already completed, this cell recovers the best model & training
# history instantly without re-training 50 epochs.

train_model = train_single_model  # alias for backwards compatibility
ckpt_path = CKPT_DIR / "best_model_ppg_vpg_apg.pt"
log_file = LOG_DIR / "training_log.txt"

if ckpt_path.exists() and log_file.exists():
    print("=" * 70)
    print("CHECKPOINT DETECTED: Loading existing trained model without retraining!")
    print(f"Source: {ckpt_path}")
    print("=" * 70)
    best_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_trained = PPGCNNBaseline().to(device)
    model_trained.load_state_dict(best_checkpoint["model_state_dict"])
    model_trained.eval()
    
    # Parse training history and loss curves from training_log.txt
    import re
    training_history, train_loss_curve, val_loss_curve = [], [], []
    pattern = re.compile(r"Epoch (\\d+)/\\d+ \\| TrainLoss=([\\d\\.]+) \\| ValLoss=([\\d\\.]+) \\| SBP=([\\d\\.]+) DBP=([\\d\\.]+) Comb=([\\d\\.]+) mmHg \\| LR=([\\d\\.e\\-\\+]+)")
    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep, t_l, v_l = int(m.group(1)), float(m.group(2)), float(m.group(3))
                sbp, dbp, comb, lr = float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7))
                train_loss_curve.append(t_l)
                val_loss_curve.append(v_l)
                training_history.append({
                    "epoch": ep, "train_loss": t_l, "val_loss": v_l,
                    "val_sbp_mae": sbp, "val_dbp_mae": dbp, "val_comb_mae": comb, "lr": lr
                })
    print(f"Successfully loaded best checkpoint from Epoch {best_checkpoint['epoch']}:")
    print(f"  Validation SBP MAE:  {best_checkpoint['val_sbp_mae']:.2f} mmHg")
    print(f"  Validation DBP MAE:  {best_checkpoint['val_dbp_mae']:.2f} mmHg")
    print(f"  Validation Comb MAE: {best_checkpoint['val_comb_mae']:.2f} mmHg")
    print(f"Recovered full {len(training_history)}-epoch loss history curve.")
else:
    model_trained, best_checkpoint, training_history, train_loss_curve, val_loss_curve = train_single_model(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
    )
""")
    cells.append(c12)

    # =========================================================================
    # Cell 13: Single Frozen Test-Set Evaluation
    # =========================================================================
    c13 = nbf.v4.new_code_cell("""# 12. Single Frozen Test-Set Evaluation & Classical Benchmark Comparison
# STRICT PROTOCOL: The unseen test set (38,361 windows) is evaluated exactly ONCE using the best validation checkpoint.

@torch.no_grad()
def evaluate_model_on_loader(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb)
        all_true.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    all_true = np.vstack(all_true)
    all_pred = np.vstack(all_pred)
    return all_true[:, 0], all_true[:, 1], all_pred[:, 0], all_pred[:, 1]

print("Evaluating frozen best model on test set (38,361 windows)...")
t_sbp_true, t_dbp_true, t_sbp_pred, t_dbp_pred = evaluate_model_on_loader(model_trained, test_loader, device)

metrics_test_sbp = compute_regression_metrics(t_sbp_true, t_sbp_pred)
metrics_test_dbp = compute_regression_metrics(t_dbp_true, t_dbp_pred)
comb_test_mae = (metrics_test_sbp["mae"] + metrics_test_dbp["mae"]) / 2.0

v_sbp_true = best_checkpoint["val_true"][:, 0]
v_dbp_true = best_checkpoint["val_true"][:, 1]
v_sbp_pred = best_checkpoint["val_pred"][:, 0]
v_dbp_pred = best_checkpoint["val_pred"][:, 1]

metrics_val_sbp = compute_regression_metrics(v_sbp_true, v_sbp_pred)
metrics_val_dbp = compute_regression_metrics(v_dbp_true, v_dbp_pred)
comb_val_mae = (metrics_val_sbp["mae"] + metrics_val_dbp["mae"]) / 2.0

# Save predictions
np.savez_compressed(PRED_DIR / "validation_predictions.npz",
    val_true=best_checkpoint["val_true"], val_pred=best_checkpoint["val_pred"])

np.savez_compressed(PRED_DIR / "test_predictions.npz",
    test_sbp_true=t_sbp_true, test_dbp_true=t_dbp_true,
    test_sbp_pred=t_sbp_pred, test_dbp_pred=t_dbp_pred)

df_test_preds = pd.DataFrame({
    "record_id": test_ds.manifest["record_id"].values,
    "window_id": test_ds.manifest["window_id"].values if "window_id" in test_ds.manifest else np.arange(len(t_sbp_true)),
    "sbp_true": t_sbp_true, "sbp_pred": t_sbp_pred,
    "dbp_true": t_dbp_true, "dbp_pred": t_dbp_pred,
})
df_test_preds.to_csv(PRED_DIR / "test_predictions.csv", index=False)

# Save metrics CSVs
df_val_metrics = pd.DataFrame([{"target": "SBP", **metrics_val_sbp}, {"target": "DBP", **metrics_val_dbp}])
df_test_metrics = pd.DataFrame([{"target": "SBP", **metrics_test_sbp}, {"target": "DBP", **metrics_test_dbp}])
df_val_metrics.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)
df_test_metrics.to_csv(METRICS_DIR / "test_metrics.csv", index=False)

# Benchmark comparison table
comp_df = pd.DataFrame([
    {
        "Model": "Phase 3A Classical Baseline (HistGB)",
        "Split": "Test (Frozen)",
        "SBP MAE (mmHg)": PHASE3A_SBP_MAE,
        "DBP MAE (mmHg)": PHASE3A_DBP_MAE,
        "Combined MAE (mmHg)": PHASE3A_COMB_MAE,
        "SBP RMSE": 18.27, "DBP RMSE": 9.42,
    },
    {
        "Model": "Phase 4A Neural Baseline (PPG+VPG+APG 1D CNN)",
        "Split": "Validation",
        "SBP MAE (mmHg)": round(metrics_val_sbp["mae"], 2),
        "DBP MAE (mmHg)": round(metrics_val_dbp["mae"], 2),
        "Combined MAE (mmHg)": round(comb_val_mae, 2),
        "SBP RMSE": round(metrics_val_sbp["rmse"], 2),
        "DBP RMSE": round(metrics_val_dbp["rmse"], 2),
    },
    {
        "Model": "Phase 4A Neural Baseline (PPG+VPG+APG 1D CNN)",
        "Split": "Test (Frozen)",
        "SBP MAE (mmHg)": round(metrics_test_sbp["mae"], 2),
        "DBP MAE (mmHg)": round(metrics_test_dbp["mae"], 2),
        "Combined MAE (mmHg)": round(comb_test_mae, 2),
        "SBP RMSE": round(metrics_test_sbp["rmse"], 2),
        "DBP RMSE": round(metrics_test_dbp["rmse"], 2),
    },
])

print("\\n" + "=" * 90)
print("PHASE 4A FINAL BENCHMARK COMPARISON TABLE")
print("=" * 90)
print(comp_df.to_string(index=False))
print("=" * 90)
print("NOTE: Descriptive numerical comparison only. No claims of clinical equivalence.")
""")
    cells.append(c13)

    # =========================================================================
    # Cell 14: Stratified Clinical BP Range Analysis
    # =========================================================================
    c14 = nbf.v4.new_code_cell("""# 13. Stratified Clinical BP Range Error Analysis
# Evaluates error breakdown across clinical ranges:
# SBP: <90, 90-119, 120-139, 140-159, >=160
# DBP: <60, 60-79, 80-89, 90-99, >=100

strat_test_sbp = compute_bp_range_stratification(t_sbp_true, t_sbp_pred, "SBP")
strat_test_dbp = compute_bp_range_stratification(t_dbp_true, t_dbp_pred, "DBP")

strat_test_sbp.to_csv(METRICS_DIR / "stratified_test_sbp.csv", index=False)
strat_test_dbp.to_csv(METRICS_DIR / "stratified_test_dbp.csv", index=False)

print("--- SBP STRATIFIED PERFORMANCE (TEST SET) ---")
print(strat_test_sbp.to_string(index=False))
print("\\n--- DBP STRATIFIED PERFORMANCE (TEST SET) ---")
print(strat_test_dbp.to_string(index=False))
""")
    cells.append(c14)

    # =========================================================================
    # Cell 15: Record-Level Test Analysis
    # =========================================================================
    c15 = nbf.v4.new_code_cell("""# 14. Record-Level Test Analysis (Record-Independent Evaluation)
# Because the split is record-level, evaluate performance aggregated by record_id.
# TERMINOLOGY: "record-independent evaluation" (subject IDs are not explicitly present).

df_rec_metrics = compute_record_level_metrics(test_ds.manifest, t_sbp_true, t_sbp_pred, t_dbp_true, t_dbp_pred)
df_rec_metrics.to_csv(METRICS_DIR / "record_level_test_metrics.csv", index=False)

def summarize_distribution(series: pd.Series, name: str) -> Dict[str, float]:
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    return {
        "Metric": name,
        "Mean": float(series.mean()),
        "Median": float(series.median()),
        "Std": float(series.std()),
        "IQR": float(q75 - q25),
        "Min": float(series.min()),
        "Max": float(series.max()),
    }

rec_summary_df = pd.DataFrame([
    summarize_distribution(df_rec_metrics["sbp_mae"], "Record-Level SBP MAE (mmHg)"),
    summarize_distribution(df_rec_metrics["dbp_mae"], "Record-Level DBP MAE (mmHg)"),
    summarize_distribution(df_rec_metrics["comb_mae"], "Record-Level Combined MAE (mmHg)"),
])

print("=" * 80)
print(f"RECORD-LEVEL EVALUATION SUMMARY ({len(df_rec_metrics):,} test records):")
print("=" * 80)
print(rec_summary_df.to_string(index=False))
print("=" * 80)
""")
    cells.append(c15)

    # =========================================================================
    # Cell 16: All 12 Publication Visualizations
    # =========================================================================
    c16 = nbf.v4.new_code_cell("""# 15. Publication Visualizations (All 12 Figures)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "figure.dpi": 130, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3
})

def plot_bland_altman(ax, y_true, y_pred, title, color="crimson"):
    diff = y_pred - y_true
    mean = (y_true + y_pred) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff))
    ax.scatter(mean, diff, s=1, alpha=0.1, color=color, rasterized=True)
    ax.axhline(bias, color="black", linestyle="-", linewidth=1.5, label=f"Bias: {bias:.2f}")
    ax.axhline(bias + 1.96*sd, color="red", linestyle="--", linewidth=1.2, label=f"+1.96 SD: {bias+1.96*sd:.2f}")
    ax.axhline(bias - 1.96*sd, color="red", linestyle="--", linewidth=1.2, label=f"-1.96 SD: {bias-1.96*sd:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Mean BP (mmHg)")
    ax.set_ylabel("Predicted - Reference (mmHg)")
    ax.legend(loc="upper right", fontsize=8)

# -------------------------------------------------------------
# FIG 1: Training vs Validation Huber Loss
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_loss_curve, label="Training Huber Loss", color="#059669", linewidth=2)
ax.plot(val_loss_curve, label="Validation Huber Loss", color="#047857", linestyle="--", linewidth=2)
ax.set_title("Phase 4A Single-Model: Training vs Validation Huber Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber Loss")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig01_training_validation_loss.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 2: Validation SBP & DBP MAE Progression
# -------------------------------------------------------------
val_sbp_maes = [h["val_sbp_mae"] for h in training_history]
val_dbp_maes = [h["val_dbp_mae"] for h in training_history]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(val_sbp_maes, label="Validation SBP MAE", color="#2563eb", linewidth=2)
ax.plot(val_dbp_maes, label="Validation DBP MAE", color="#10b981", linewidth=2)
ax.axhline(PHASE3A_SBP_MAE, color="#93c5fd", linestyle="--", label=f"Phase 3A SBP Baseline ({PHASE3A_SBP_MAE:.2f})")
ax.axhline(PHASE3A_DBP_MAE, color="#6ee7b7", linestyle="--", label=f"Phase 3A DBP Baseline ({PHASE3A_DBP_MAE:.2f})")
ax.set_title("Validation MAE Progression Across Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig02_val_mae_evolution.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 3: Test SBP Parity Scatter Plot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(t_sbp_true, t_sbp_pred, s=1, alpha=0.12, color="#2563eb", rasterized=True)
ax.plot([50, 200], [50, 200], "k--", label="Identity (y = x)")
ax.set_title(f"Phase 4A Test SBP Parity (MAE={metrics_test_sbp['mae']:.2f}, R²={metrics_test_sbp['r2']:.3f})")
ax.set_xlabel("Reference SBP (mmHg)")
ax.set_ylabel("Predicted SBP (mmHg)")
ax.set_xlim(50, 200)
ax.set_ylim(50, 200)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig03_test_sbp_scatter.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 4: Test DBP Parity Scatter Plot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(t_dbp_true, t_dbp_pred, s=1, alpha=0.12, color="#10b981", rasterized=True)
ax.plot([30, 130], [30, 130], "k--", label="Identity (y = x)")
ax.set_title(f"Phase 4A Test DBP Parity (MAE={metrics_test_dbp['mae']:.2f}, R²={metrics_test_dbp['r2']:.3f})")
ax.set_xlabel("Reference DBP (mmHg)")
ax.set_ylabel("Predicted DBP (mmHg)")
ax.set_xlim(30, 130)
ax.set_ylim(30, 130)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig04_test_dbp_scatter.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 5: Test SBP Bland-Altman Agreement
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
plot_bland_altman(ax, t_sbp_true, t_sbp_pred, "Phase 4A Test SBP Bland-Altman Agreement", color="#2563eb")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig05_test_sbp_bland_altman.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 6: Test DBP Bland-Altman Agreement
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
plot_bland_altman(ax, t_dbp_true, t_dbp_pred, "Phase 4A Test DBP Bland-Altman Agreement", color="#10b981")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig06_test_dbp_bland_altman.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 7: SBP Error by Clinical BP Range
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(strat_test_sbp["range"], strat_test_sbp["mae"], color="#2563eb", width=0.5)
ax.axhline(metrics_test_sbp["mae"], color="red", linestyle="--", label=f"Overall Test MAE ({metrics_test_sbp['mae']:.2f})")
ax.set_title("Test SBP MAE Stratified by Clinical BP Range")
ax.set_xlabel("SBP Clinical Range (mmHg)")
ax.set_ylabel("MAE (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig07_sbp_range_error.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 8: DBP Error by Clinical BP Range
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(strat_test_dbp["range"], strat_test_dbp["mae"], color="#10b981", width=0.5)
ax.axhline(metrics_test_dbp["mae"], color="red", linestyle="--", label=f"Overall Test MAE ({metrics_test_dbp['mae']:.2f})")
ax.set_title("Test DBP MAE Stratified by Clinical BP Range")
ax.set_xlabel("DBP Clinical Range (mmHg)")
ax.set_ylabel("MAE (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig08_dbp_range_error.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 9: SBP Error Residual Histogram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
err_sbp = t_sbp_pred - t_sbp_true
ax.hist(err_sbp, bins=100, range=(-40, 40), density=True, color="#2563eb", alpha=0.7, label=f"SBP Error (Mean={np.mean(err_sbp):.2f}, SD={np.std(err_sbp):.2f})")
ax.set_title("Phase 4A Test SBP Error Residual Distribution")
ax.set_xlabel("Predicted - Reference SBP (mmHg)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig09_sbp_error_distribution.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 10: DBP Error Residual Histogram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
err_dbp = t_dbp_pred - t_dbp_true
ax.hist(err_dbp, bins=100, range=(-30, 30), density=True, color="#10b981", alpha=0.7, label=f"DBP Error (Mean={np.mean(err_dbp):.2f}, SD={np.std(err_dbp):.2f})")
ax.set_title("Phase 4A Test DBP Error Residual Distribution")
ax.set_xlabel("Predicted - Reference DBP (mmHg)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig10_dbp_error_distribution.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 11: Representative Test 10s Window (PPG, VPG, APG)
# -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
sample_x, sample_y = test_ds[0]
sample_x = sample_x.numpy()
t_axis = np.linspace(0, 10.0, WINDOW_SAMPLES)

ax1.plot(t_axis, sample_x[0], color="#2563eb", linewidth=1.2)
ax1.set_title(f"Representative 10-Second Test Window (Reference SBP={sample_y[0]:.1f}, DBP={sample_y[1]:.1f} mmHg) - Normalized PPG")
ax1.set_ylabel("PPG (z-norm)")

ax2.plot(t_axis, sample_x[1], color="#d97706", linewidth=1.2)
ax2.set_title("First Time Derivative: Velocity Plethysmogram (VPG)")
ax2.set_ylabel("VPG (z-norm)")

ax3.plot(t_axis, sample_x[2], color="#dc2626", linewidth=1.2)
ax3.set_title("Second Time Derivative: Acceleration Plethysmogram (APG)")
ax3.set_xlabel("Time (seconds)")
ax3.set_ylabel("APG (z-norm)")

plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig11_sample_10s_waveform.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# FIG 12: Record-Level MAE Distribution
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(df_rec_metrics["sbp_mae"], bins=50, range=(0, 35), color="#2563eb", alpha=0.7, label="Record SBP MAE")
ax1.axvline(df_rec_metrics["sbp_mae"].median(), color="black", linestyle="--", label=f"Median: {df_rec_metrics['sbp_mae'].median():.2f}")
ax1.set_title("Record-Level SBP MAE Distribution")
ax1.set_xlabel("MAE (mmHg)")
ax1.set_ylabel("Record Count")
ax1.legend()

ax2.hist(df_rec_metrics["dbp_mae"], bins=50, range=(0, 20), color="#10b981", alpha=0.7, label="Record DBP MAE")
ax2.axvline(df_rec_metrics["dbp_mae"].median(), color="black", linestyle="--", label=f"Median: {df_rec_metrics['dbp_mae'].median():.2f}")
ax2.set_title("Record-Level DBP MAE Distribution")
ax2.set_xlabel("MAE (mmHg)")
ax2.set_ylabel("Record Count")
ax2.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig12_record_level_mae_distribution.png"), dpi=300)
plt.show()

print(f"All 12 single-model figures generated and saved to: {FIG_DIR}")
""")
    cells.append(c16)

    # =========================================================================
    # Cell 17: Scientific Report & Evidence Freeze
    # =========================================================================
    c17 = nbf.v4.new_code_cell("""# 16. Scientific Reports & Evidence Freeze
# Generates PHASE4A_SINGLE_MODEL_REPORT.md, PHASE4A_SINGLE_MODEL_EVIDENCE_FREEZE.md, and metadata JSON.

best_ep = best_checkpoint["epoch"]

report_content = f\"\"\"# PHASE 4A — Single-Model Neural Baseline Report

**Architecture:** PPG + VPG + APG 1D CNN  
**Mode:** Calibration-Free Cuffless Blood Pressure Estimation  
**Execution Environment:** Local System ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Main Research Objective
Can a compact 3-channel neural representation consisting of PPG, VPG, and APG estimate SBP and DBP from a single 10-second PPG window under the existing calibration-free, record-level leakage-controlled protocol?

## 2. Dataset and Protocol
- **Dataset Source:** PhysioNet MIMIC-II / Kachuee Blood Pressure Dataset (12 MAT parts)
- **Record-level Partition:** 8,400 train records, 1,800 validation records, 1,800 test records
- **Eligible Windows:** Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}
- **Sampling Frequency:** Fs = 125 Hz (1,250 samples per 10-second non-overlapping window)

## 3. Strict Sensor Rules & Leakage Controls
- Channel 0 (PPG) was used strictly as model input.
- Channel 1 (ABP) was used strictly as reference ground truth target.
- Channel 2 (ECG) was strictly forbidden and never accessed.
- Record IDs between Train, Validation, and Test sets were strictly pairwise disjoint (0 record leakage).

## 4. Preprocessing & Input Definitions
- 3rd-order zero-phase Butterworth bandpass filter (0.5 - 8.0 Hz)
- Velocity Plethysmogram (VPG): 1st time derivative
- Acceleration Plethysmogram (APG): 2nd time derivative
- Per-window z-score normalization across all 3 channels
- Input tensor shape: [3, 1250], float32

## 5. Neural Architecture & Hyperparameters
- Architecture: 4 ConvBlocks (Conv1D + BN + ReLU + MaxPool/AdaptiveAvgPool) -> FC(128 -> 64) -> Dual Linear Heads (SBP, DBP)
- Total Parameters: 146,978 (trainable: 146,978)
- Loss: Huber loss (delta=5.0) summed across SBP and DBP
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Batch size: {BATCH_SIZE}
- Early stopping patience: {EARLY_STOP_PATIENCE} epochs

## 6. Benchmark Performance Comparison (Frozen Test Set)

| Model / Architecture | Split | SBP MAE (mmHg) | DBP MAE (mmHg) | Combined MAE (mmHg) | SBP RMSE | DBP RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 3A Classical Baseline (HistGB)** | Test (Frozen) | 13.93 | 7.05 | 10.49 | 18.27 | 9.42 |
| **Phase 4A Neural Baseline (PPG+VPG+APG)** | Val | {metrics_val_sbp['mae']:.2f} | {metrics_val_dbp['mae']:.2f} | {comb_val_mae:.2f} | {metrics_val_sbp['rmse']:.2f} | {metrics_val_dbp['rmse']:.2f} |
| **Phase 4A Neural Baseline (PPG+VPG+APG)** | Test (Frozen) | {metrics_test_sbp['mae']:.2f} | {metrics_test_dbp['mae']:.2f} | {comb_test_mae:.2f} | {metrics_test_sbp['rmse']:.2f} | {metrics_test_dbp['rmse']:.2f} |

## 7. Clinical BP Range Stratified Errors (Descriptive)

### SBP Ranges:
{strat_test_sbp.to_markdown(index=False)}

### DBP Ranges:
{strat_test_dbp.to_markdown(index=False)}

## 8. Record-Level Performance (Record-Independent Evaluation)
- SBP Record MAE: Mean = {df_rec_metrics['sbp_mae'].mean():.2f}, Median = {df_rec_metrics['sbp_mae'].median():.2f}, SD = {df_rec_metrics['sbp_mae'].std():.2f} mmHg
- DBP Record MAE: Mean = {df_rec_metrics['dbp_mae'].mean():.2f}, Median = {df_rec_metrics['dbp_mae'].median():.2f}, SD = {df_rec_metrics['dbp_mae'].std():.2f} mmHg
- Combined Record MAE: Mean = {df_rec_metrics['comb_mae'].mean():.2f}, Median = {df_rec_metrics['comb_mae'].median():.2f}, SD = {df_rec_metrics['comb_mae'].std():.2f} mmHg

## 9. Hardware Deployment Note
Future deployment on MAX30102 (100 Hz) utilizes polyphase anti-aliasing resampling:
`ppg_125hz = resample_poly(ppg_100hz, up=5, down=4)`
Research training was conducted strictly at 125 Hz.

## 10. Scientific Limitations & Next Research Step
- Evaluated on ICU patient records; performance on ambulatory healthy subjects remains unverified.
- Phase 4A establishes the compact single-window neural baseline.
- Next Step (Phase 4B): Introduce temporal sequence context (GRU/LSTM/Transformer) over successive windows.
\"\"\"

with open(REPORT_DIR / "PHASE4A_SINGLE_MODEL_REPORT.md", "w") as f:
    f.write(report_content)

# Evidence Freeze Document
freeze_content = f\"\"\"# PHASE 4A SINGLE-MODEL EVIDENCE FREEZE

Only one neural model was trained in Phase 4A:
PPG + VPG + APG 1D CNN.

- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset Path:** {DATASET_DIR}
- **Manifest Path:** {manifest_path}
- **Split Records:** Train = {len(train_recs)}, Val = {len(val_recs)}, Test = {len(test_recs)}
- **Eligible Windows:** Train = {len(train_ds)}, Val = {len(val_ds)}, Test = {len(test_ds)}
- **Sampling Rate:** Fs = 125 Hz (1,250 samples / 10s)
- **Input Channels:** 3 (PPG, VPG, APG)
- **Architecture:** PPGCNNBaseline (146,978 parameters)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Batch Size:** {BATCH_SIZE}
- **Max Epochs:** {MAX_EPOCHS} (Early stopping patience = {EARLY_STOP_PATIENCE})
- **Random Seed:** {SEED}
- **Best Validation Epoch:** {best_ep}
- **Best Validation SBP MAE:** {metrics_val_sbp['mae']:.4f} mmHg
- **Best Validation DBP MAE:** {metrics_val_dbp['mae']:.4f} mmHg
- **Best Validation Combined MAE:** {comb_val_mae:.4f} mmHg
- **Final Test SBP MAE:** {metrics_test_sbp['mae']:.4f} mmHg (RMSE={metrics_test_sbp['rmse']:.4f}, R2={metrics_test_sbp['r2']:.4f})
- **Final Test DBP MAE:** {metrics_test_dbp['mae']:.4f} mmHg (RMSE={metrics_test_dbp['rmse']:.4f}, R2={metrics_test_dbp['r2']:.4f})
- **Final Test Combined MAE:** {comb_test_mae:.4f} mmHg
- **Checkpoint Path:** {CKPT_DIR / 'best_model_ppg_vpg_apg.pt'}
- **Environment:** Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda if torch.cuda.is_available() else 'None'}
\"\"\"

with open(REPORT_DIR / "PHASE4A_SINGLE_MODEL_EVIDENCE_FREEZE.md", "w") as f:
    f.write(freeze_content)

# Metadata JSON
meta_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "experiment": "Phase 4A Single-Model (PPG + VPG + APG 1D CNN)",
    "architecture": "PPGCNNBaseline",
    "total_parameters": 146978,
    "input_channels": 3,
    "sampling_rate_hz": FS,
    "window_samples": WINDOW_SAMPLES,
    "batch_size": BATCH_SIZE,
    "huber_delta": HUBER_DELTA,
    "best_epoch": best_ep,
    "validation_metrics": {
        "sbp": metrics_val_sbp,
        "dbp": metrics_val_dbp,
        "combined_mae": comb_val_mae,
    },
    "test_metrics": {
        "sbp": metrics_test_sbp,
        "dbp": metrics_test_dbp,
        "combined_mae": comb_test_mae,
    },
    "phase3a_baseline": {
        "sbp_mae": PHASE3A_SBP_MAE,
        "dbp_mae": PHASE3A_DBP_MAE,
        "combined_mae": PHASE3A_COMB_MAE,
    },
    "environment": {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    }
}

with open(REPORT_DIR / "phase4a_single_model_metadata.json", "w") as f:
    json.dump(meta_data, f, indent=2)

print("=" * 70)
print(f"Report saved:   {REPORT_DIR / 'PHASE4A_SINGLE_MODEL_REPORT.md'}")
print(f"Freeze saved:   {REPORT_DIR / 'PHASE4A_SINGLE_MODEL_EVIDENCE_FREEZE.md'}")
print(f"Metadata saved: {REPORT_DIR / 'phase4a_single_model_metadata.json'}")
print("=" * 70)
print("PHASE 4A SINGLE-MODEL WORKFLOW COMPLETE.")
""")
    cells.append(c17)

    nb.cells = cells

    out_path = Path("code/notebooks/04A_neural_ppg_vpg_apg_single_model.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated clean single-model notebook at: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    generate_single_model_notebook()
