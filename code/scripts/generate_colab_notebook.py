"""
Script to generate the refined self-contained Google Colab notebook for Phase 4A.
Generates: code/notebooks/04A_neural_ppg_baseline_colab.ipynb
"""

import json
from pathlib import Path
import nbformat as nbf

def create_colab_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    }

    cells = []

    # -------------------------------------------------------------
    # Cell 1: Markdown Header
    # -------------------------------------------------------------
    c1 = nbf.v4.new_markdown_cell("""# PHASE 4A — Neural PPG Baseline
## 10-second PPG + VPG + APG 1D CNN for Calibration-Free Cuffless Blood Pressure Estimation

**Target Research Question:**
> *Can a compact neural representation of a single 10-second PPG waveform, together with deterministic VPG and APG derivative channels, outperform the existing frozen classical PPG-only baseline?*

---

### Key Experimental Constraints & Protocols:
1. **Inputs**: Strictly 10-second PPG waveforms ($F_s = 125$ Hz, 1,250 samples).
   - **ECG (channel 2) is strictly forbidden** as model input.
   - **ABP (channel 1) is used strictly as ground truth target**, never input.
2. **Calibration-Free**: Zero subject-specific baseline BP tuning or recalibration.
3. **Controlled Ablation**:
   - **Model A**: 1 Channel (PPG only).
   - **Model B**: 3 Channels (PPG + VPG + APG).
   - Identical architecture, optimizer, learning rate, loss function, and early stopping.
4. **Frozen Baselines (Phase 3A Classical HistGB)**:
   - SBP MAE: **13.93 mmHg**
   - DBP MAE: **7.05 mmHg**
   - Combined MAE: **10.49 mmHg**
5. **Leakage Control**: Disjoint record-level partition (Train: 8,400 records, Val: 1,800 records, Test: 1,800 records).

---
### Runtime Instructions:
1. Ensure your Colab runtime is set to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU` or better).
2. Run each cell sequentially.
3. Download the final `phase4a_results.zip` package containing checkpoints, predictions, figures, and reports.
""")
    cells.append(c1)

    # -------------------------------------------------------------
    # Cell 2: Setup Environment & Dependencies
    # -------------------------------------------------------------
    c2 = nbf.v4.new_code_cell("""# Environment & GPU Verification
import os
import sys
import time
import gc
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Print hardware & software environment
print("=" * 70)
print(f"Python Version:  {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    device = torch.device("cuda")
else:
    print("WARNING: CUDA is not available. Please switch runtime to GPU!")
    device = torch.device("cpu")
print("=" * 70)

# Reproducibility seeds
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
print(f"Deterministic random seed set to {SEED}.")
""")
    cells.append(c2)

    # -------------------------------------------------------------
    # Cell 3: Directory Configuration & Data Options
    # -------------------------------------------------------------
    c3 = nbf.v4.new_code_cell("""# Workspace Directory Layout
BASE_DIR = Path("/content/phase4a_workspace")
DATASET_DIR = BASE_DIR / "BloodPressureDataset"
OUTPUT_DIR = BASE_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

for d in [BASE_DIR, DATASET_DIR, CKPT_DIR, METRICS_DIR, PRED_DIR, FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Workspace initialized at: {BASE_DIR}")
""")
    cells.append(c3)

    # -------------------------------------------------------------
    # Cell 4: Dataset Acquisition (Active Kaggle Download)
    # -------------------------------------------------------------
    c4 = nbf.v4.new_code_cell("""# DATASET ACQUISITION
# Method A: Fast Kaggle API Download (~1-2 minutes in Colab at 80+ MB/s)
import os
os.environ['KAGGLE_USERNAME'] = "omkarphadatare"
os.environ['KAGGLE_KEY'] = "KGAT_19529c0d6c2b1d22954bf9d20316d26b"

!pip install -q kaggle
!kaggle datasets download -d mkachuee/BloodPressureDataset -p {DATASET_DIR} --unzip

# Verify MAT files presence
mat_files = sorted(list(DATASET_DIR.glob("part_*.mat")))
if len(mat_files) == 0:
    alt_candidates = [Path("/content"), Path("/content/data"), Path("/content/BloodPressureDataset")]
    for alt in alt_candidates:
        cand_mats = sorted(list(alt.glob("part_*.mat")))
        if len(cand_mats) > 0:
            DATASET_DIR = alt
            mat_files = cand_mats
            print(f"Found MAT files in alternative directory: {DATASET_DIR}")
            break

print(f"Found {len(mat_files)} MAT file(s) in {DATASET_DIR}:")
for f in mat_files[:5]:
    print(f"  - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")
if len(mat_files) > 5:
    print(f"  ... and {len(mat_files) - 5} more.")

if len(mat_files) == 0:
    print("\\n[!] No MAT files detected yet. Please verify Kaggle download above.")
""")
    cells.append(c4)

    # -------------------------------------------------------------
    # Cell 5: Window Manifest & Leakage Verification
    # -------------------------------------------------------------
    c5 = nbf.v4.new_code_cell("""# ACQUIRE & VALIDATE WINDOW MANIFEST
# The manifest contains 261,658 physiological windows across 12,000 records.
# It can be loaded from window_manifest.csv or window_manifest.csv.gz (only 20 MB).

manifest_path = None
candidates = [
    BASE_DIR / "window_manifest.csv.gz",
    BASE_DIR / "window_manifest.csv",
    Path("/content/window_manifest.csv.gz"),
    Path("/content/window_manifest.csv"),
    Path("/content/phase4a_workspace/window_manifest.csv.gz"),
    Path("/content/phase4a_workspace/window_manifest.csv"),
    Path("/content/drive/MyDrive/window_manifest.csv.gz"),
    Path("/content/drive/MyDrive/window_manifest.csv"),
]
for c in candidates:
    if c.exists():
        manifest_path = c
        break

if manifest_path is None:
    print("=" * 70)
    print("MANIFEST FILE NOT DETECTED IN WORKSPACE YET.")
    print("Please upload window_manifest.csv.gz (20 MB).")
    print("File path on your machine:")
    print("  code/outputs/windows/window_manifest.csv.gz")
    print("=" * 70)
    try:
        from google.colab import files
        uploaded = files.upload()
        for fname in uploaded:
            if "manifest" in fname:
                manifest_path = BASE_DIR / fname
                with open(manifest_path, "wb") as f:
                    f.write(uploaded[fname])
                break
    except Exception as e:
        print(f"Note on file upload: {e}")
        print("If using VS Code Colab extension, drag and drop 'window_manifest.csv.gz'")
        print("directly into the '/content/phase4a_workspace' folder in your file sidebar.")

if manifest_path is None or not Path(manifest_path).exists():
    raise FileNotFoundError(
        "Manifest file not found. Please upload window_manifest.csv.gz to /content/phase4a_workspace "
        "or run files.upload() above."
    )

print(f"Reading manifest from: {manifest_path}...")
df_full = pd.read_csv(manifest_path)
df_eligible = df_full[df_full["modeling_eligible"] == True].copy().reset_index(drop=True)

# -------------------------------------------------------------------------
# STRICT LEAKAGE CONTROL ASSERTIONS
# -------------------------------------------------------------------------
train_recs = set(df_eligible[df_eligible["split"] == "train"]["record_id"].unique())
val_recs = set(df_eligible[df_eligible["split"] == "val"]["record_id"].unique())
test_recs = set(df_eligible[df_eligible["split"] == "test"]["record_id"].unique())

assert len(train_recs & val_recs) == 0, "CRITICAL ERROR: Leakage between Train and Val sets!"
assert len(train_recs & test_recs) == 0, "CRITICAL ERROR: Leakage between Train and Test sets!"
assert len(val_recs & test_recs) == 0, "CRITICAL ERROR: Leakage between Val and Test sets!"

print("=" * 60)
print("ZERO-LEAKAGE RECORD PARTITION VERIFIED:")
print(f"  Train: {len(train_recs):>5} records | {len(df_eligible[df_eligible['split'] == 'train']):>7} eligible windows")
print(f"  Val:   {len(val_recs):>5} records | {len(df_eligible[df_eligible['split'] == 'val']):>7} eligible windows")
print(f"  Test:  {len(test_recs):>5} records | {len(df_eligible[df_eligible['split'] == 'test']):>7} eligible windows")
print(f"  Total: {len(train_recs)+len(val_recs)+len(test_recs):>5} records | {len(df_eligible):>7} eligible windows")
print("=" * 60)
""")
    cells.append(c5)

    # -------------------------------------------------------------
    # Cell 6: Preprocessing Module
    # -------------------------------------------------------------
    c6 = nbf.v4.new_code_cell("""# PREPROCESSING MODULE
# Standardized signal processing:
# 1. Zero-phase Butterworth bandpass filter (0.5 - 8.0 Hz, 3rd order offline filtfilt)
# 2. VPG (Velocity Plethysmogram) = 1st derivative (np.gradient)
# 3. APG (Acceleration Plethysmogram) = 2nd derivative (np.gradient)
# 4. Per-window z-score normalization: (x - mean) / (std + 1e-8)

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

def preprocess_window(raw_ppg: np.ndarray, n_channels: int = 3, use_zscore: bool = True) -> np.ndarray:
    raw_ppg = np.asarray(raw_ppg, dtype=np.float64)
    if len(raw_ppg) != WINDOW_SAMPLES:
        raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {len(raw_ppg)}")
    
    dt = 1.0 / FS
    ppg_filt = bandpass_filter(raw_ppg, fs=FS)
    
    if n_channels == 1:
        ch_ppg = per_window_zscore(ppg_filt) if use_zscore else ppg_filt
        return np.expand_dims(ch_ppg, axis=0).astype(np.float32)
    
    vpg = compute_vpg(ppg_filt, dt=dt)
    apg = compute_apg(vpg, dt=dt)
    
    if use_zscore:
        ppg_filt = per_window_zscore(ppg_filt)
        vpg = per_window_zscore(vpg)
        apg = per_window_zscore(apg)
        
    return np.stack([ppg_filt, vpg, apg], axis=0).astype(np.float32)

print("Preprocessing functions compiled successfully.")
""")
    cells.append(c6)

    # -------------------------------------------------------------
    # Cell 7: PyTorch Dataset
    # -------------------------------------------------------------
    c7 = nbf.v4.new_code_cell("""# PYTORCH DATASET WITH ZERO LEAKAGE & OPTIMIZED MEMORY REUSE
# Loads PPG waveforms grouped by MAT file parts to prevent disk thrashing.
# STRICT PROTOCOL: Channel 0 is PPG. Channel 2 (ECG) is NEVER accessed.

PPG_CHANNEL_IDX = 0

class PPGWindowDataset(Dataset):
    def __init__(
        self,
        df_manifest: pd.DataFrame,
        dataset_dir: Path,
        split: str,
        n_channels: int = 3,
        use_zscore: bool = True,
        max_windows: Optional[int] = None,
        cached_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        assert n_channels in (1, 3), f"n_channels must be 1 or 3, got {n_channels}"
        self.split = split
        self.n_channels = n_channels
        self.use_zscore = use_zscore
        self.dataset_dir = Path(dataset_dir)
        
        # Filter split
        df_split = df_manifest[df_manifest["split"] == split].copy().reset_index(drop=True)
        if max_windows is not None:
            df_split = df_split.iloc[:max_windows].copy().reset_index(drop=True)
            
        self.manifest = df_split
        
        if cached_data is not None:
            self._ppg_data, self._sbp, self._dbp = cached_data
            print(f"[{split.upper()}] Reusing preloaded raw buffer ({len(self._ppg_data):,} windows, instant!).")
        else:
            print(f"[{split.upper()}] Loading {len(df_split):,} windows from MAT files...")
            self._ppg_data, self._sbp, self._dbp = self._load_windows(df_split)
            print(f"[{split.upper()}] Memory buffer ready: {self._ppg_data.shape}, dtype={self._ppg_data.dtype}")

    def _load_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        ppg_array = np.empty((n, WINDOW_SAMPLES), dtype=np.float32)
        sbp_array = df["sbp"].values.astype(np.float32)
        dbp_array = df["dbp"].values.astype(np.float32)
        
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
                # STRICT: Channel 0 is PPG. ECG (channel 2) is NEVER indexed.
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
        x = preprocess_window(raw_ppg, n_channels=self.n_channels, use_zscore=self.use_zscore)
        y = np.array([self._sbp[idx], self._dbp[idx]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

print("PPGWindowDataset class defined.")
""")
    cells.append(c7)

    # -------------------------------------------------------------
    # Cell 8: 1D CNN Model Architecture
    # -------------------------------------------------------------
    c8 = nbf.v4.new_code_cell("""# 1D CNN MODEL ARCHITECTURE (PPGCNNBaseline)
# 4 ConvBlocks with BatchNorm, ReLU, and Pooling -> AdaptiveAvgPool1d(1) -> FC Dual Heads

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
    def __init__(self, n_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        assert n_channels in (1, 3)
        self.n_channels = n_channels
        
        # 4 Convolutional Feature Extraction Blocks
        self.block1 = ConvBlock(n_channels, 32, kernel_size=7, pool_size=2)
        self.block2 = ConvBlock(32, 64, kernel_size=7, pool_size=2)
        self.block3 = ConvBlock(64, 128, kernel_size=5, pool_size=2)
        self.block4 = ConvBlock(128, 128, kernel_size=5, use_adaptive_pool=True)
        
        # Dense Regression Heads
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
        return torch.cat([sbp, dbp], dim=1)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# Instantiate and verify parameter counts
model_a_demo = PPGCNNBaseline(n_channels=1)
model_b_demo = PPGCNNBaseline(n_channels=3)
print(f"Model A (PPG-only, 1 channel):     {model_a_demo.count_parameters()[0]:,} parameters")
print(f"Model B (PPG+VPG+APG, 3 channels): {model_b_demo.count_parameters()[0]:,} parameters")
""")
    cells.append(c8)

    # -------------------------------------------------------------
    # Cell 9: Smoke Test
    # -------------------------------------------------------------
    c9 = nbf.v4.new_code_cell("""# SMOKE TEST EXECUTION
# Verifies forward pass, backward pass, finite gradients, and loss reduction on 128 windows.

def run_smoke_test(device, manifest_df, dataset_dir):
    print("=" * 60)
    print("RUNNING PIPELINE SMOKE TEST")
    print("=" * 60)
    set_seed(42)
    
    ds_smoke = PPGWindowDataset(
        manifest_df, dataset_dir, split="train",
        n_channels=3, use_zscore=True, max_windows=128
    )
    loader_smoke = DataLoader(ds_smoke, batch_size=32, shuffle=True)
    
    model = PPGCNNBaseline(n_channels=3).to(device)
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

# Run smoke test
run_smoke_test(device, df_eligible, DATASET_DIR)
""")
    cells.append(c9)

    # -------------------------------------------------------------
    # Cell 10: Training Function & Metric Utilities
    # -------------------------------------------------------------
    c10 = nbf.v4.new_code_cell("""# TRAINING LOOP & METRICS ENGINE
HUBER_DELTA = 5.0
BATCH_SIZE = 256
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = len(y_true)
    var_true = float(np.var(y_true))
    r2 = float(1.0 - np.sum(errors ** 2) / (n * var_true)) if var_true > 1e-9 else 0.0
    
    # BHS Standard percentages (<= 5, 10, 15 mmHg)
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

def compute_bp_range_errors(y_true: np.ndarray, y_pred: np.ndarray, target: str = "SBP") -> pd.DataFrame:
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
            "target": target, "range": lbl, "count": cnt,
            "mae": float(np.mean(abs_err[m])) if cnt > 0 else np.nan,
            "rmse": float(np.sqrt(np.mean(err[m]**2))) if cnt > 0 else np.nan,
            "bias": float(np.mean(err[m])) if cnt > 0 else np.nan,
            "sd": float(np.std(err[m])) if cnt > 0 else np.nan,
        })
    return pd.DataFrame(rows)

def train_model(
    model_name: str,
    n_channels: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
):
    set_seed(42)
    print("=" * 70)
    print(f"TRAINING: {model_name} (n_channels={n_channels})")
    print("=" * 70)
    
    model = PPGCNNBaseline(n_channels=n_channels).to(device)
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
    ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
    
    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        # 1. Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred[:, 0], yb[:, 0]) + criterion(pred[:, 1], yb[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # 2. Validate
        model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
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
            }, ckpt_path)
            mark = "BEST *"
        else:
            no_improve += 1
            mark = f"NoImprove={no_improve}"
            
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | TrainLoss={avg_train:.3f} | ValLoss={avg_val:.3f} | "
              f"SBP={sbp_mae:.2f} DBP={dbp_mae:.2f} Comb={comb_mae:.2f} mmHg | LR={curr_lr:.1e} | {mark}")
        
        history.append({
            "epoch": epoch, "train_loss": avg_train, "val_loss": avg_val,
            "val_sbp_mae": sbp_mae, "val_dbp_mae": dbp_mae, "val_comb_mae": comb_mae, "lr": curr_lr
        })
        
        if no_improve >= patience:
            print(f"\\nEarly stopping triggered after {patience} epochs without improvement.")
            break
            
    elapsed = (time.time() - start_time) / 60.0
    print(f"\\nTraining finished in {elapsed:.1f} min. Best model at epoch {best_epoch} with Val Comb MAE={best_val_comb:.2f} mmHg.")
    
    # Reload best weights
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model, best_ckpt, history, train_losses, val_losses

print("Training engine compiled.")
""")
    cells.append(c10)

    # -------------------------------------------------------------
    # Cell 11: Data Loaders for Full Run
    # -------------------------------------------------------------
    c11 = nbf.v4.new_code_cell("""# BUILD SPLIT DATASETS & LOADERS
# Load raw waveforms once, then reuse across Model A (1 ch) and Model B (3 ch) for maximum speed.

print("Initializing Split Datasets...")
train_ds_a = PPGWindowDataset(df_eligible, DATASET_DIR, split="train", n_channels=1)
val_ds_a   = PPGWindowDataset(df_eligible, DATASET_DIR, split="val", n_channels=1)
test_ds_a  = PPGWindowDataset(df_eligible, DATASET_DIR, split="test", n_channels=1)

train_loader_a = DataLoader(train_ds_a, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True)
val_loader_a   = DataLoader(val_ds_a, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
test_loader_a  = DataLoader(test_ds_a, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

# Re-use preloaded raw arrays for Model B (instantly ready with 0 extra disk I/O):
train_ds_b = PPGWindowDataset(df_eligible, DATASET_DIR, split="train", n_channels=3,
                              cached_data=(train_ds_a._ppg_data, train_ds_a._sbp, train_ds_a._dbp))
val_ds_b   = PPGWindowDataset(df_eligible, DATASET_DIR, split="val", n_channels=3,
                              cached_data=(val_ds_a._ppg_data, val_ds_a._sbp, val_ds_a._dbp))
test_ds_b  = PPGWindowDataset(df_eligible, DATASET_DIR, split="test", n_channels=3,
                              cached_data=(test_ds_a._ppg_data, test_ds_a._sbp, test_ds_a._dbp))

train_loader_b = DataLoader(train_ds_b, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True)
val_loader_b   = DataLoader(val_ds_b, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
test_loader_b  = DataLoader(test_ds_b, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

print("\\nAll DataLoaders initialized successfully.")
""")
    cells.append(c11)

    # -------------------------------------------------------------
    # Cell 12: Train Model A
    # -------------------------------------------------------------
    c12 = nbf.v4.new_code_cell("""# TRAIN MODEL A (PPG-ONLY BASELINE: 1 CHANNEL)
model_a, ckpt_a, hist_a, train_losses_a, val_losses_a = train_model(
    model_name="model_a_ppg_only",
    n_channels=1,
    train_loader=train_loader_a,
    val_loader=val_loader_a,
    device=device,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOP_PATIENCE
)
""")
    cells.append(c12)

    # -------------------------------------------------------------
    # Cell 13: Train Model B
    # -------------------------------------------------------------
    c13 = nbf.v4.new_code_cell("""# TRAIN MODEL B (PPG + VPG + APG BASELINE: 3 CHANNELS)
model_b, ckpt_b, hist_b, train_losses_b, val_losses_b = train_model(
    model_name="model_b_ppg_vpg_apg",
    n_channels=3,
    train_loader=train_loader_b,
    val_loader=val_loader_b,
    device=device,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOP_PATIENCE
)
""")
    cells.append(c13)

    # -------------------------------------------------------------
    # Cell 14: Single Frozen Test-Set Evaluation
    # -------------------------------------------------------------
    c14 = nbf.v4.new_code_cell("""# SINGLE FROZEN TEST-SET EVALUATION
# Protocol strictly requires evaluating each frozen best checkpoint ONCE on the unseen test set.

@torch.no_grad()
def evaluate_test_set(model, loader, device):
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

print("Evaluating Model A on Test Set (38,361 windows)...")
t_sbp_true, t_dbp_true, t_sbp_pred_a, t_dbp_pred_a = evaluate_test_set(model_a, test_loader_a, device)
metrics_test_a_sbp = compute_metrics(t_sbp_true, t_sbp_pred_a)
metrics_test_a_dbp = compute_metrics(t_dbp_true, t_dbp_pred_a)

print("Evaluating Model B on Test Set (38,361 windows)...")
_, _, t_sbp_pred_b, t_dbp_pred_b = evaluate_test_set(model_b, test_loader_b, device)
metrics_test_b_sbp = compute_metrics(t_sbp_true, t_sbp_pred_b)
metrics_test_b_dbp = compute_metrics(t_dbp_true, t_dbp_pred_b)

# Val metrics from best checkpoints
v_sbp_true, v_dbp_true = ckpt_a["val_true"][:, 0], ckpt_a["val_true"][:, 1]
v_sbp_pred_a, v_dbp_pred_a = ckpt_a["val_pred"][:, 0], ckpt_a["val_pred"][:, 1]
v_sbp_pred_b, v_dbp_pred_b = ckpt_b["val_pred"][:, 0], ckpt_b["val_pred"][:, 1]

metrics_val_a_sbp = compute_metrics(v_sbp_true, v_sbp_pred_a)
metrics_val_a_dbp = compute_metrics(v_dbp_true, v_dbp_pred_a)
metrics_val_b_sbp = compute_metrics(v_sbp_true, v_sbp_pred_b)
metrics_val_b_dbp = compute_metrics(v_dbp_true, v_dbp_pred_b)

# Save predictions
np.savez_compressed(PRED_DIR / "model_a_predictions.npz",
    val_true=ckpt_a["val_true"], val_pred=ckpt_a["val_pred"],
    test_sbp_true=t_sbp_true, test_dbp_true=t_dbp_true,
    test_sbp_pred=t_sbp_pred_a, test_dbp_pred=t_dbp_pred_a)

np.savez_compressed(PRED_DIR / "model_b_predictions.npz",
    val_true=ckpt_b["val_true"], val_pred=ckpt_b["val_pred"],
    test_sbp_true=t_sbp_true, test_dbp_true=t_dbp_true,
    test_sbp_pred=t_sbp_pred_b, test_dbp_pred=t_dbp_pred_b)

# Display Comparison Summary Table
comb_val_a = (metrics_val_a_sbp["mae"] + metrics_val_a_dbp["mae"]) / 2
comb_val_b = (metrics_val_b_sbp["mae"] + metrics_val_b_dbp["mae"]) / 2
comb_test_a = (metrics_test_a_sbp["mae"] + metrics_test_a_dbp["mae"]) / 2
comb_test_b = (metrics_test_b_sbp["mae"] + metrics_test_b_dbp["mae"]) / 2

summary_df = pd.DataFrame([
    {"Model": "Phase 3A Classical Baseline (HistGB)", "Split": "Test", "SBP MAE": 13.93, "DBP MAE": 7.05, "Combined MAE": 10.49, "SBP RMSE": 18.27, "DBP RMSE": 9.42},
    {"Model": "Model A: Neural PPG-Only (1-Ch CNN)", "Split": "Val", "SBP MAE": metrics_val_a_sbp["mae"], "DBP MAE": metrics_val_a_dbp["mae"], "Combined MAE": comb_val_a, "SBP RMSE": metrics_val_a_sbp["rmse"], "DBP RMSE": metrics_val_a_dbp["rmse"]},
    {"Model": "Model A: Neural PPG-Only (1-Ch CNN)", "Split": "Test", "SBP MAE": metrics_test_a_sbp["mae"], "DBP MAE": metrics_test_a_dbp["mae"], "Combined MAE": comb_test_a, "SBP RMSE": metrics_test_a_sbp["rmse"], "DBP RMSE": metrics_test_a_dbp["rmse"]},
    {"Model": "Model B: Neural PPG+VPG+APG (3-Ch CNN)", "Split": "Val", "SBP MAE": metrics_val_b_sbp["mae"], "DBP MAE": metrics_val_b_dbp["mae"], "Combined MAE": comb_val_b, "SBP RMSE": metrics_val_b_sbp["rmse"], "DBP RMSE": metrics_val_b_dbp["rmse"]},
    {"Model": "Model B: Neural PPG+VPG+APG (3-Ch CNN)", "Split": "Test", "SBP MAE": metrics_test_b_sbp["mae"], "DBP MAE": metrics_test_b_dbp["mae"], "Combined MAE": comb_test_b, "SBP RMSE": metrics_test_b_sbp["rmse"], "DBP RMSE": metrics_test_b_dbp["rmse"]},
])

print("\\n" + "=" * 90)
print("PHASE 4A FINAL BENCHMARK COMPARISON TABLE")
print("=" * 90)
print(summary_df.to_string(index=False))
print("=" * 90)
""")
    cells.append(c14)

    # -------------------------------------------------------------
    # Cell 15: Stratified BP Range Error Analysis
    # -------------------------------------------------------------
    c15 = nbf.v4.new_code_cell("""# STRATIFIED CLINICAL BP RANGE ERROR ANALYSIS
# Evaluates error breakdown across Hypotensive, Normal, Prehypertensive, Stage 1, Stage 2

strat_test_a_sbp = compute_bp_range_errors(t_sbp_true, t_sbp_pred_a, "SBP")
strat_test_a_dbp = compute_bp_range_errors(t_dbp_true, t_dbp_pred_a, "DBP")
strat_test_b_sbp = compute_bp_range_errors(t_sbp_true, t_sbp_pred_b, "SBP")
strat_test_b_dbp = compute_bp_range_errors(t_dbp_true, t_dbp_pred_b, "DBP")

strat_test_a_sbp.to_csv(METRICS_DIR / "stratified_test_a_sbp.csv", index=False)
strat_test_b_sbp.to_csv(METRICS_DIR / "stratified_test_b_sbp.csv", index=False)

print("--- SBP STRATIFIED ERROR (TEST SET) ---")
comp_sbp_strat = pd.DataFrame({
    "Range": strat_test_a_sbp["range"],
    "Sample Count": strat_test_a_sbp["count"],
    "Model A SBP MAE": strat_test_a_sbp["mae"].round(2),
    "Model B SBP MAE": strat_test_b_sbp["mae"].round(2),
    "Delta (B - A)": (strat_test_b_sbp["mae"] - strat_test_a_sbp["mae"]).round(2),
})
print(comp_sbp_strat.to_string(index=False))

print("\\n--- DBP STRATIFIED ERROR (TEST SET) ---")
comp_dbp_strat = pd.DataFrame({
    "Range": strat_test_a_dbp["range"],
    "Sample Count": strat_test_a_dbp["count"],
    "Model A DBP MAE": strat_test_a_dbp["mae"].round(2),
    "Model B DBP MAE": strat_test_b_dbp["mae"].round(2),
    "Delta (B - A)": (strat_test_b_dbp["mae"] - strat_test_a_dbp["mae"]).round(2),
})
print(comp_dbp_strat.to_string(index=False))
""")
    cells.append(c15)

    # -------------------------------------------------------------
    # Cell 16: Publication Figures (12 Figures)
    # -------------------------------------------------------------
    c16 = nbf.v4.new_code_cell("""# PUBLICATION VISUALIZATIONS (12 FIGURES)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "figure.dpi": 130, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3
})

def plot_bland_altman(ax, y_true, y_pred, title, color="crimson"):
    diff = y_pred - y_true
    mean = (y_true + y_pred) / 2.0
    bias = np.mean(diff)
    sd = np.std(diff)
    ax.scatter(mean, diff, s=1, alpha=0.1, color=color, rasterized=True)
    ax.axhline(bias, color="black", linestyle="-", linewidth=1.5, label=f"Bias: {bias:.2f}")
    ax.axhline(bias + 1.96*sd, color="red", linestyle="--", linewidth=1.2, label=f"+1.96 SD: {bias+1.96*sd:.2f}")
    ax.axhline(bias - 1.96*sd, color="red", linestyle="--", linewidth=1.2, label=f"-1.96 SD: {bias-1.96*sd:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Mean BP (mmHg)")
    ax.set_ylabel("Predicted - Reference (mmHg)")
    ax.legend(loc="upper right", fontsize=8)

# FIG 1 & 2: Training Loss & Validation MAE Progression
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses_a, label="Train Loss (Model A)", color="#2563eb")
ax1.plot(val_losses_a, label="Val Loss (Model A)", color="#1d4ed8", linestyle="--")
ax1.plot(train_losses_b, label="Train Loss (Model B)", color="#059669")
ax1.plot(val_losses_b, label="Val Loss (Model B)", color="#047857", linestyle="--")
ax1.set_title("Training & Validation Huber Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Huber Loss")
ax1.legend()

val_sbp_maes_a = [h["val_sbp_mae"] for h in hist_a]
val_dbp_maes_a = [h["val_dbp_mae"] for h in hist_a]
val_sbp_maes_b = [h["val_sbp_mae"] for h in hist_b]
val_dbp_maes_b = [h["val_dbp_mae"] for h in hist_b]

ax2.plot(val_sbp_maes_a, label="Model A SBP MAE", color="#2563eb")
ax2.plot(val_dbp_maes_a, label="Model A DBP MAE", color="#93c5fd", linestyle=":")
ax2.plot(val_sbp_maes_b, label="Model B SBP MAE", color="#059669")
ax2.plot(val_dbp_maes_b, label="Model B DBP MAE", color="#6ee7b7", linestyle=":")
ax2.axhline(13.93, color="gray", linestyle="--", label="Phase 3A SBP (13.93)")
ax2.axhline(7.05, color="lightgray", linestyle="--", label="Phase 3A DBP (7.05)")
ax2.set_title("Validation MAE Progression Across Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MAE (mmHg)")
ax2.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig01_fig02_training_validation_curves.png"), dpi=300)
plt.show()

# FIG 3 & 4: Test Bland-Altman for Model A
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_bland_altman(ax1, t_sbp_true, t_sbp_pred_a, "Model A Test SBP Bland-Altman", color="#2563eb")
plot_bland_altman(ax2, t_dbp_true, t_dbp_pred_a, "Model A Test DBP Bland-Altman", color="#3b82f6")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig03_fig04_model_a_bland_altman.png"), dpi=300)
plt.show()

# FIG 5 & 6: Test Bland-Altman for Model B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_bland_altman(ax1, t_sbp_true, t_sbp_pred_b, "Model B Test SBP Bland-Altman", color="#059669")
plot_bland_altman(ax2, t_dbp_true, t_dbp_pred_b, "Model B Test DBP Bland-Altman", color="#10b981")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig05_fig06_model_b_bland_altman.png"), dpi=300)
plt.show()

# FIG 7 & 8: Parity Scatter Plots (Test Set)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.scatter(t_sbp_true, t_sbp_pred_b, s=1, alpha=0.1, color="#059669", rasterized=True)
ax1.plot([50, 200], [50, 200], "k--", label="y = x")
ax1.set_title(f"Model B Test SBP: MAE={metrics_test_b_sbp['mae']:.2f}, R2={metrics_test_b_sbp['r2']:.3f}")
ax1.set_xlabel("Reference SBP (mmHg)")
ax1.set_ylabel("Predicted SBP (mmHg)")
ax1.legend()

ax2.scatter(t_dbp_true, t_dbp_pred_b, s=1, alpha=0.1, color="#10b981", rasterized=True)
ax2.plot([30, 130], [30, 130], "k--", label="y = x")
ax2.set_title(f"Model B Test DBP: MAE={metrics_test_b_dbp['mae']:.2f}, R2={metrics_test_b_dbp['r2']:.3f}")
ax2.set_xlabel("Reference DBP (mmHg)")
ax2.set_ylabel("Predicted DBP (mmHg)")
ax2.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig07_fig08_model_b_parity_scatters.png"), dpi=300)
plt.show()

# FIG 9 & 10: Error Residual Distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(t_sbp_pred_a - t_sbp_true, bins=80, range=(-40, 40), density=True, alpha=0.5, color="#2563eb", label="Model A")
ax1.hist(t_sbp_pred_b - t_sbp_true, bins=80, range=(-40, 40), density=True, alpha=0.5, color="#059669", label="Model B")
ax1.set_title("Test SBP Error Distribution (Pred - True)")
ax1.set_xlabel("Error (mmHg)")
ax1.legend()

ax2.hist(t_dbp_pred_a - t_dbp_true, bins=80, range=(-30, 30), density=True, alpha=0.5, color="#2563eb", label="Model A")
ax2.hist(t_dbp_pred_b - t_dbp_true, bins=80, range=(-30, 30), density=True, alpha=0.5, color="#059669", label="Model B")
ax2.set_title("Test DBP Error Distribution (Pred - True)")
ax2.set_xlabel("Error (mmHg)")
ax2.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig09_fig10_error_distributions.png"), dpi=300)
plt.show()

# FIG 11 & 12: Stratified Clinical BP Range Error Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x_idx = np.arange(len(comp_sbp_strat))
w = 0.35
ax1.bar(x_idx - w/2, comp_sbp_strat["Model A SBP MAE"], width=w, color="#2563eb", label="Model A (PPG-Only)")
ax1.bar(x_idx + w/2, comp_sbp_strat["Model B SBP MAE"], width=w, color="#059669", label="Model B (PPG+VPG+APG)")
ax1.set_xticks(x_idx)
ax1.set_xticklabels(comp_sbp_strat["Range"])
ax1.set_title("Test SBP MAE Across Clinical BP Ranges")
ax1.set_xlabel("SBP Range (mmHg)")
ax1.set_ylabel("MAE (mmHg)")
ax1.legend()

x_idx_d = np.arange(len(comp_dbp_strat))
ax2.bar(x_idx_d - w/2, comp_dbp_strat["Model A DBP MAE"], width=w, color="#2563eb", label="Model A (PPG-Only)")
ax2.bar(x_idx_d + w/2, comp_dbp_strat["Model B DBP MAE"], width=w, color="#059669", label="Model B (PPG+VPG+APG)")
ax2.set_xticks(x_idx_d)
ax2.set_xticklabels(comp_dbp_strat["Range"])
ax2.set_title("Test DBP MAE Across Clinical BP Ranges")
ax2.set_xlabel("DBP Range (mmHg)")
ax2.set_ylabel("MAE (mmHg)")
ax2.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig11_fig12_stratified_bp_range_errors.png"), dpi=300)
plt.show()

print(f"All 12 publication figures generated and saved to: {FIG_DIR}")
""")
    cells.append(c16)

    # -------------------------------------------------------------
    # Cell 17: Scientific Report Generation
    # -------------------------------------------------------------
    c17 = nbf.v4.new_code_cell("""# GENERATE SCIENTIFIC REPORTS
# Compiles PHASE4A_NEURAL_BASELINE_REPORT.md and metadata JSON

report_md = f\"\"\"# PHASE 4A — Neural PPG Baseline Report
**Architecture:** 10-second PPG (+ VPG + APG) 1D CNN  
**Mode:** Calibration-Free Cuffless Blood Pressure Estimation  
**Execution Environment:** Google Colab GPU  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Research Summary

Can a compact neural representation of a single 10-second PPG waveform outperform the classical baseline?

| Model / Architecture | Channels | Split | SBP MAE (mmHg) | DBP MAE (mmHg) | Combined MAE (mmHg) | SBP RMSE | DBP RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 3A Classical (HistGB)** | 1 | Test | 13.93 | 7.05 | 10.49 | 18.27 | 9.42 |
| **Model A: Neural PPG-Only** | 1 | Val | {metrics_val_a_sbp['mae']:.2f} | {metrics_val_a_dbp['mae']:.2f} | {comb_val_a:.2f} | {metrics_val_a_sbp['rmse']:.2f} | {metrics_val_a_dbp['rmse']:.2f} |
| **Model A: Neural PPG-Only** | 1 | Test | {metrics_test_a_sbp['mae']:.2f} | {metrics_test_a_dbp['mae']:.2f} | {comb_test_a:.2f} | {metrics_test_a_sbp['rmse']:.2f} | {metrics_test_a_dbp['rmse']:.2f} |
| **Model B: Neural PPG+VPG+APG** | 3 | Val | {metrics_val_b_sbp['mae']:.2f} | {metrics_val_b_dbp['mae']:.2f} | {comb_val_b:.2f} | {metrics_val_b_sbp['rmse']:.2f} | {metrics_val_b_dbp['rmse']:.2f} |
| **Model B: Neural PPG+VPG+APG** | 3 | Test | {metrics_test_b_sbp['mae']:.2f} | {metrics_test_b_dbp['mae']:.2f} | {comb_test_b:.2f} | {metrics_test_b_sbp['rmse']:.2f} | {metrics_test_b_dbp['rmse']:.2f} |

---

## 2. Clinical Standard Compliance (Test Set)

### British Hypertension Society (BHS) Standard
| Model | Target | <= 5 mmHg (%) | <= 10 mmHg (%) | <= 15 mmHg (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Model A** | SBP | {metrics_test_a_sbp['pct_within_5']:.1f}% | {metrics_test_a_sbp['pct_within_10']:.1f}% | {metrics_test_a_sbp['pct_within_15']:.1f}% |
| **Model A** | DBP | {metrics_test_a_dbp['pct_within_5']:.1f}% | {metrics_test_a_dbp['pct_within_10']:.1f}% | {metrics_test_a_dbp['pct_within_15']:.1f}% |
| **Model B** | SBP | {metrics_test_b_sbp['pct_within_5']:.1f}% | {metrics_test_b_sbp['pct_within_10']:.1f}% | {metrics_test_b_sbp['pct_within_15']:.1f}% |
| **Model B** | DBP | {metrics_test_b_dbp['pct_within_5']:.1f}% | {metrics_test_b_dbp['pct_within_10']:.1f}% | {metrics_test_b_dbp['pct_within_15']:.1f}% |

### AAMI Standard Evaluation (Criteria: Mean Error <= 5 mmHg, SD <= 8 mmHg)
- **Model A SBP:** Bias = {metrics_test_a_sbp['bias']:.2f} mmHg, SD = {metrics_test_a_sbp['error_sd']:.2f} mmHg
- **Model A DBP:** Bias = {metrics_test_a_dbp['bias']:.2f} mmHg, SD = {metrics_test_a_dbp['error_sd']:.2f} mmHg
- **Model B SBP:** Bias = {metrics_test_b_sbp['bias']:.2f} mmHg, SD = {metrics_test_b_sbp['error_sd']:.2f} mmHg
- **Model B DBP:** Bias = {metrics_test_b_dbp['bias']:.2f} mmHg, SD = {metrics_test_b_dbp['error_sd']:.2f} mmHg

---

## 3. Scientific Finding & Phase 4B Decision
- Controlled ablation strictly isolates the effect of derivative channels (VPG and APG).
- Single frozen test evaluation performed exactly once with zero leakage.
\"\"\"

with open(REPORT_DIR / "PHASE4A_NEURAL_BASELINE_REPORT.md", "w") as f:
    f.write(report_md)

# Metadata JSON
meta_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "device": str(device),
    "model_a": {
        "val_sbp_mae": metrics_val_a_sbp["mae"],
        "val_dbp_mae": metrics_val_a_dbp["mae"],
        "test_sbp_mae": metrics_test_a_sbp["mae"],
        "test_dbp_mae": metrics_test_a_dbp["mae"],
    },
    "model_b": {
        "val_sbp_mae": metrics_val_b_sbp["mae"],
        "val_dbp_mae": metrics_val_b_dbp["mae"],
        "test_sbp_mae": metrics_test_b_sbp["mae"],
        "test_dbp_mae": metrics_test_b_dbp["mae"],
    },
    "phase3a_baseline": {
        "sbp_mae": 13.93,
        "dbp_mae": 7.05,
        "comb_mae": 10.49,
    }
}
with open(REPORT_DIR / "phase4a_training_metadata.json", "w") as f:
    json.dump(meta_data, f, indent=2)

print(f"Report successfully saved to: {REPORT_DIR / 'PHASE4A_NEURAL_BASELINE_REPORT.md'}")
""")
    cells.append(c17)

    # -------------------------------------------------------------
    # Cell 18: Package & Download All Artifacts
    # -------------------------------------------------------------
    c18 = nbf.v4.new_code_cell("""# ONE-CLICK PACKAGE & DOWNLOAD RESULTS
import zipfile
from pathlib import Path

zip_output_path = BASE_DIR / "phase4a_results.zip"
print(f"Packaging all checkpoints, predictions, figures, and reports into: {zip_output_path}...")

with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for folder in [CKPT_DIR, METRICS_DIR, PRED_DIR, FIG_DIR, REPORT_DIR]:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(OUTPUT_DIR)
                zf.write(file_path, arcname=str(rel_path))

print(f"Archive created! Size: {zip_output_path.stat().st_size / (1024**2):.2f} MB")
print(f"File location on Colab: {zip_output_path}")

try:
    from google.colab import files
    print("Attempting automatic browser download...")
    files.download(str(zip_output_path))
except Exception as e:
    print(f"Note: Automatic browser download skipped ({e}).")
    print("If running in VS Code, right-click on 'phase4a_results.zip' in the Colab file tree to download.")
""")
    cells.append(c18)

    nb.cells = cells

    out_path = Path("code/notebooks/04A_neural_ppg_baseline_colab.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated Colab notebook at: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    create_colab_notebook()
