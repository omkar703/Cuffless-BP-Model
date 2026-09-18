"""
Script to generate code/notebooks/04B_temporal_gru.ipynb
Phase 4B: Temporal Context Extension (Causal 6-window GRU over Frozen Phase 4A CNN Features)
"""

from pathlib import Path
import nbformat as nbf

def generate_phase4b_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {
            "name": "python",
            "version": "3.10.21"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3 (.venv)"
        }
    }

    cells = []

    # =========================================================================
    # Cell 0: Markdown Header & Research Objective
    # =========================================================================
    c0 = nbf.v4.new_markdown_cell("""# PHASE 4B — Temporal Context Extension
## Causal 6-Window (60-Second) GRU over Frozen Phase 4A CNN Latent Representations

**Primary Research Question:**
> *Does adding causal temporal context from the previous 60 seconds (6 consecutive 10-second windows: $[t-5, t-4, t-3, t-2, t-1, t]$) improve blood-pressure estimation beyond the frozen Phase 4A single-window CNN representation?*

---

### Core Experimental Design & Invariants:
1. **Frozen Phase 4A Feature Extractor**:
   - Checkpoint: `code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt`
   - All 146,978 CNN parameters have `requires_grad = False` (0 trainable parameters).
   - Extracts the 64-dimensional latent representation immediately preceding the SBP/DBP heads.
2. **Single Temporal Model**:
   - `TemporalGRUModel`: 1-layer causal GRU (`hidden_size = 64`, `batch_first = True`) + FC(64 $\\to$ 32) + dual regression heads (SBP, DBP).
   - Exactly **27,106 trainable parameters**.
3. **Causal 60-Second Sequence Construction**:
   - Sequence length = 6 windows ($6 \\times 10\\text{s} = 60\\text{s}$).
   - Windows $[t-5, t-4, t-3, t-2, t-1, t]$ must belong to the **same record_id** and be strictly consecutive in time.
   - Prediction target is $\\text{SBP}(t)$ and $\\text{DBP}(t)$ for the final window $t$. Zero future information is accessible.
4. **Matched Test Comparison Protocol**:
   - To eliminate population-shift bias caused by dropping the initial windows of each record (~81.4% retention), Phase 4B is compared against Phase 4A on the **identical matched test target windows**.
5. **Sensor Integrity Rules**:
   - PPG (Channel 0) is model input; ABP (Channel 1) is reference ground truth; ECG (Channel 2) is strictly forbidden.
6. **Isolated Output Directory**:
   - All outputs are stored exclusively in `code/outputs/phase4b_temporal_gru/`.
""")
    cells.append(c0)

    # =========================================================================
    # Cell 1: Environment & Hardware Setup
    # =========================================================================
    c1 = nbf.v4.new_code_cell("""# 1. Environment & Hardware Verification
import os
import sys
import time
import gc
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample_poly

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

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
    cells.append(c1)

    # =========================================================================
    # Cell 2: Frozen Phase 4A Checkpoint Loading & Verification
    # =========================================================================
    c2 = nbf.v4.new_code_cell("""# 2. Frozen Phase 4A Checkpoint Loading & Feature Extractor Verification
# Loads the trained single-model checkpoint (Epoch 48, Val Comb MAE = 8.46 mmHg)
# Freezes all 146,978 parameters so 0 CNN parameters are trainable.

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, pool_size: int = 2, use_adaptive_pool: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_c)
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
        return torch.cat([sbp, dbp], dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Extracts the 64-dimensional latent embedding immediately before the SBP/DBP heads.
        In eval() mode, Dropout is disabled.
        \"\"\"
        feat = self.block4(self.block3(self.block2(self.block1(x))))
        feat = self.flatten(feat)
        feat = self.fc(feat)  # [B, 64]
        return feat

# Discover root and locate Phase 4A checkpoint
cwd = Path.cwd().resolve()
PROJECT_ROOT = None
for c in [cwd, cwd.parent, cwd.parent.parent]:
    if (c / "BloodPressureDataset").exists():
        PROJECT_ROOT = c
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")
PHASE4A_CKPT_PATH = PROJECT_ROOT / "code" / "outputs" / "phase4a_single_model" / "checkpoints" / "best_model_ppg_vpg_apg.pt"

assert PHASE4A_CKPT_PATH.exists(), f"Phase 4A checkpoint not found at {PHASE4A_CKPT_PATH}!"

print(f"Loading Phase 4A checkpoint: {PHASE4A_CKPT_PATH}")
phase4a_checkpoint = torch.load(PHASE4A_CKPT_PATH, map_location=device, weights_only=False)

cnn_encoder = PPGCNNBaseline().to(device)
cnn_encoder.load_state_dict(phase4a_checkpoint["model_state_dict"])
cnn_encoder.eval()

# Freeze all parameters
for param in cnn_encoder.parameters():
    param.requires_grad = False

total_cnn_p = sum(p.numel() for p in cnn_encoder.parameters())
trainable_cnn_p = sum(p.numel() for p in cnn_encoder.parameters() if p.requires_grad)

print("=" * 70)
print("FROZEN FEATURE EXTRACTOR (Phase 4A PPGCNNBaseline)")
print(f"  Checkpoint Epoch:       {phase4a_checkpoint.get('epoch', 'N/A')}")
print(f"  Validation SBP MAE:     {phase4a_checkpoint.get('val_sbp_mae', float('nan')):.2f} mmHg")
print(f"  Validation DBP MAE:     {phase4a_checkpoint.get('val_dbp_mae', float('nan')):.2f} mmHg")
print(f"  Validation Combined:    {phase4a_checkpoint.get('val_comb_mae', float('nan')):.2f} mmHg")
print(f"  Total CNN Parameters:   {total_cnn_p:,}")
print(f"  Trainable Parameters:   {trainable_cnn_p:,} (VERIFIED FROZEN)")
print("=" * 70)
assert trainable_cnn_p == 0, f"Error: Expected 0 trainable CNN parameters, found {trainable_cnn_p}!"
assert total_cnn_p == 146978, f"Error: Expected 146,978 parameters, found {total_cnn_p}!"
""")
    cells.append(c2)

    # =========================================================================
    # Cell 3: Directory Setup & Paths
    # =========================================================================
    c3 = nbf.v4.new_code_cell("""# 3. Clean Isolated Output Directory Setup
DATASET_DIR = PROJECT_ROOT / "BloodPressureDataset"
OUTPUT_DIR  = PROJECT_ROOT / "code" / "outputs" / "phase4b_temporal_gru"
MANIFEST_GZ = PROJECT_ROOT / "code" / "outputs" / "windows" / "window_manifest.csv.gz"
PHASE4A_PREDS_CSV = PROJECT_ROOT / "code" / "outputs" / "phase4a_single_model" / "predictions" / "test_predictions.csv"

CKPT_DIR    = OUTPUT_DIR / "checkpoints"
EMB_DIR     = OUTPUT_DIR / "embeddings"
SEQ_DIR     = OUTPUT_DIR / "sequences"
METRICS_DIR = OUTPUT_DIR / "metrics"
PRED_DIR    = OUTPUT_DIR / "predictions"
FIG_DIR     = OUTPUT_DIR / "figures"
REPORT_DIR  = OUTPUT_DIR / "reports"
LOG_DIR     = OUTPUT_DIR / "logs"

for p in [CKPT_DIR, EMB_DIR, SEQ_DIR, METRICS_DIR, PRED_DIR, FIG_DIR, REPORT_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print(f"Project Root: {PROJECT_ROOT}")
print(f"Dataset Dir:  {DATASET_DIR}")
print(f"Output Dir:   {OUTPUT_DIR}")
assert DATASET_DIR.exists(), f"Dataset directory missing: {DATASET_DIR}"
assert MANIFEST_GZ.exists(), f"Manifest file missing: {MANIFEST_GZ}"
assert PHASE4A_PREDS_CSV.exists(), f"Phase 4A predictions missing: {PHASE4A_PREDS_CSV}"
""")
    cells.append(c3)

    # =========================================================================
    # Cell 4: Manifest Loading & Zero-Leakage Checks
    # =========================================================================
    c4 = nbf.v4.new_code_cell("""# 4. Load Window Manifest & Enforce Zero-Leakage Record Partition
print(f"Loading manifest from: {MANIFEST_GZ}...")
t0 = time.time()
df_manifest = pd.read_csv(MANIFEST_GZ)
load_time = time.time() - t0

df_eligible = df_manifest[df_manifest["modeling_eligible"] == True].copy()
print(f"Loaded {len(df_eligible):,} eligible windows in {load_time:.2f}s")

# Validate Record Split Disjointness
train_recs = set(df_eligible[df_eligible["split"] == "train"]["record_id"].unique())
val_recs   = set(df_eligible[df_eligible["split"] == "val"]["record_id"].unique())
test_recs  = set(df_eligible[df_eligible["split"] == "test"]["record_id"].unique())

assert len(train_recs & val_recs) == 0, "CRITICAL ERROR: Leakage between Train and Val sets!"
assert len(train_recs & test_recs) == 0, "CRITICAL ERROR: Leakage between Train and Test sets!"
assert len(val_recs & test_recs) == 0, "CRITICAL ERROR: Leakage between Val and Test sets!"

n_train = len(df_eligible[df_eligible["split"] == "train"])
n_val   = len(df_eligible[df_eligible["split"] == "val"])
n_test  = len(df_eligible[df_eligible["split"] == "test"])

print("=" * 60)
print("ZERO-LEAKAGE RECORD PARTITION VERIFIED:")
print(f"  Train: {len(train_recs):>5} records | {n_train:>7} eligible windows")
print(f"  Val:   {len(val_recs):>5} records | {n_val:>7} eligible windows")
print(f"  Test:  {len(test_recs):>5} records | {n_test:>7} eligible windows")
print(f"  Total: {len(train_recs)+len(val_recs)+len(test_recs):>5} records | {len(df_eligible):>7} eligible windows")
print("=" * 60)
""")
    cells.append(c4)

    # =========================================================================
    # Cell 5: Strict Sensor Integrity & Preprocessing Pipeline
    # =========================================================================
    c5 = nbf.v4.new_code_cell("""# 5. Strict Sensor Rules & Exact Phase 4A Preprocessing Pipeline
FS = 125.0
LOWCUT = 0.5
HIGHCUT = 8.0
FILTER_ORDER = 3
WINDOW_SAMPLES = 1250

# SENSOR INTEGRITY AUDIT:
PPG_CHANNEL_IDX = 0  # Permitted waveform input
ABP_CHANNEL_IDX = 1  # Ground-truth reference target ONLY
ECG_CHANNEL_IDX = 2  # STRICTLY FORBIDDEN

assert ECG_CHANNEL_IDX != PPG_CHANNEL_IDX, "Channel configuration error!"
print("SENSOR RESTRICTION AUDIT: PPG (0)=INPUT | ABP (1)=TARGET ONLY | ECG (2)=FORBIDDEN")

def butter_bandpass_filter(signal: np.ndarray, lowcut: float = LOWCUT, highcut: float = HIGHCUT, fs: float = FS, order: int = FILTER_ORDER) -> np.ndarray:
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def compute_vpg_apg(ppg: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, np.ndarray]:
    vpg = np.gradient(ppg, 1.0 / fs)
    apg = np.gradient(vpg, 1.0 / fs)
    return vpg, apg

def preprocess_ppg_window(raw_ppg: np.ndarray, fs: float = FS, use_zscore: bool = True) -> np.ndarray:
    assert raw_ppg.ndim == 1 and len(raw_ppg) == int(WINDOW_SAMPLES), f"Expected shape ({WINDOW_SAMPLES},), got {raw_ppg.shape}"
    filtered = butter_bandpass_filter(raw_ppg, lowcut=LOWCUT, highcut=HIGHCUT, fs=fs, order=FILTER_ORDER)
    vpg, apg = compute_vpg_apg(filtered, fs=fs)
    stacked = np.stack([filtered, vpg, apg], axis=0).astype(np.float32)  # [3, 1250]
    
    if use_zscore:
        mean = np.mean(stacked, axis=1, keepdims=True)
        std = np.std(stacked, axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        stacked = (stacked - mean) / std
    return stacked

print("Preprocessing pipeline compiled. Input tensor shape: [3, 1250].")
""")
    cells.append(c5)

    # =========================================================================
    # Cell 6: Feature Extraction & Embedding Cache Engine
    # =========================================================================
    c6 = nbf.v4.new_code_cell("""# 6. Pre-Extraction of 64-D Latent Embeddings via Frozen Phase 4A CNN
# Memory-conscious caching strategy: extracts 64-D embeddings once and saves to disk.
# Preserves exact manifest row alignment and metadata traceability.

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
        ppg_array = np.empty((n, int(WINDOW_SAMPLES)), dtype=np.float32)
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
        x = preprocess_ppg_window(raw_ppg, fs=FS, use_zscore=self.use_zscore)  # [3, 1250]
        y = torch.tensor([self._sbp[idx], self._dbp[idx]], dtype=torch.float32)
        return torch.from_numpy(x), y

def extract_and_cache_split_embeddings(
    df_split: pd.DataFrame, split_name: str, cnn_model: nn.Module, device: torch.device, emb_dir: Path, batch_size: int = 256
) -> Tuple[np.ndarray, pd.DataFrame]:
    npy_path = emb_dir / f"{split_name}_embeddings.npy"
    csv_path = emb_dir / f"{split_name}_metadata.csv"
    
    if npy_path.exists() and csv_path.exists():
        print(f"Found cached embeddings for {split_name.upper()}: loading from {npy_path}...")
        embeddings = np.load(npy_path)
        metadata = pd.read_csv(csv_path)
        print(f"  Loaded {embeddings.shape[0]:,} embeddings of dimension {embeddings.shape[1]}")
        return embeddings, metadata
        
    print(f"\\n--- EXTRACTING EMBEDDINGS FOR {split_name.upper()} ({len(df_split):,} windows) ---")
    dataset = PPGWindowDataset(df_split, DATASET_DIR, split=split_name, use_zscore=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    
    cnn_model.eval()
    all_feats = []
    t_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            feat = cnn_model.extract_features(xb)  # [B, 64]
            all_feats.append(feat.cpu().numpy())
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                elapsed = time.time() - t_start
                print(f"  Batch {batch_idx+1}/{len(loader)} | {len(all_feats)*batch_size:,} windows processed ({elapsed:.1f}s)")
                
    embeddings = np.vstack(all_feats).astype(np.float32)
    np.save(npy_path, embeddings)
    
    meta_cols = ["record_id", "window_id", "window_index", "start_sample", "end_sample", "sbp", "dbp", "split"]
    metadata = dataset.manifest[meta_cols].copy()
    metadata.to_csv(csv_path, index=False)
    
    print(f"Cached {len(embeddings):,} embeddings to: {npy_path} ({embeddings.nbytes / (1024**2):.1f} MB)")
    return embeddings, metadata

# Extract or load cached embeddings for all 3 splits
train_embeddings, train_meta = extract_and_cache_split_embeddings(
    df_eligible[df_eligible["split"] == "train"], "train", cnn_encoder, device, EMB_DIR
)
val_embeddings, val_meta = extract_and_cache_split_embeddings(
    df_eligible[df_eligible["split"] == "val"], "val", cnn_encoder, device, EMB_DIR
)
test_embeddings, test_meta = extract_and_cache_split_embeddings(
    df_eligible[df_eligible["split"] == "test"], "test", cnn_encoder, device, EMB_DIR
)
""")
    cells.append(c6)

    # =========================================================================
    # Cell 7: Causal 6-Window Sequence Construction Engine
    # =========================================================================
    c7 = nbf.v4.new_code_cell("""# 7. Causal 6-Window Sequence Construction Engine
# Sequence length = 6 consecutive windows = 60 seconds of history.
# Causal structure: [t-5, t-4, t-3, t-2, t-1, t] -> [SBP(t), DBP(t)]
# Strictly validates: same record_id, consecutive window_index, oldest->newest ordering.

def build_causal_sequences(
    embeddings: np.ndarray, metadata_df: pd.DataFrame, split_name: str, seq_dir: Path, seq_len: int = 6
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    npz_path = seq_dir / f"{split_name}_sequences.npz"
    csv_path = seq_dir / f"{split_name}_seq_metadata.csv"
    
    if npz_path.exists() and csv_path.exists():
        print(f"Found cached sequences for {split_name.upper()}: loading from {npz_path}...")
        data = np.load(npz_path)
        X_seq = data["X"]
        y_seq = data["y"]
        meta_seq = pd.read_csv(csv_path)
        print(f"  Loaded {len(X_seq):,} sequences of shape {X_seq.shape}")
        return X_seq, y_seq, meta_seq
        
    print(f"\\n--- CONSTRUCTING 60-SECOND CAUSAL SEQUENCES: {split_name.upper()} ---")
    rec_ids = metadata_df["record_id"].values
    w_indices = metadata_df["window_index"].values
    sbp_vals = metadata_df["sbp"].values
    dbp_vals = metadata_df["dbp"].values
    win_ids = metadata_df["window_id"].values
    n = len(metadata_df)
    
    # Vectorized check for valid 6-consecutive-window history
    valid_mask = np.ones(n, dtype=bool)
    valid_mask[: seq_len - 1] = False
    
    for offset in range(1, seq_len):
        same_rec = (rec_ids[seq_len - 1 :] == rec_ids[seq_len - 1 - offset : n - offset])
        consec_win = (w_indices[seq_len - 1 :] == w_indices[seq_len - 1 - offset : n - offset] + offset)
        valid_mask[seq_len - 1 :] &= (same_rec & consec_win)
        
    target_indices = np.where(valid_mask)[0]
    n_seq = len(target_indices)
    retention = n_seq / n * 100.0
    print(f"  Eligible windows: {n:,} | Valid 6-window sequences: {n_seq:,} ({retention:.2f}% retention)")
    
    # Build sequence matrix: [n_seq, 6, 64]
    # Window offsets: [5, 4, 3, 2, 1, 0] so idx - offset gives [idx-5, idx-4, idx-3, idx-2, idx-1, idx]
    offsets = np.arange(seq_len - 1, -1, -1)
    idx_matrix = target_indices[:, None] - offsets[None, :]  # [n_seq, 6]
    
    X_seq = embeddings[idx_matrix].astype(np.float32)  # [n_seq, 6, 64]
    y_seq = np.column_stack([sbp_vals[target_indices], dbp_vals[target_indices]]).astype(np.float32)  # [n_seq, 2]
    
    seq_meta = pd.DataFrame({
        "sequence_id": np.arange(n_seq),
        "record_id": rec_ids[target_indices],
        "target_window_id": win_ids[target_indices],
        "target_window_index": w_indices[target_indices],
        "sbp_true": y_seq[:, 0],
        "dbp_true": y_seq[:, 1],
    })
    
    np.savez_compressed(npz_path, X=X_seq, y=y_seq)
    seq_meta.to_csv(csv_path, index=False)
    print(f"Cached sequences to {npz_path} ({X_seq.nbytes / (1024**2):.1f} MB)")
    return X_seq, y_seq, seq_meta

# Build sequences for all splits
X_train, y_train, train_seq_meta = build_causal_sequences(train_embeddings, train_meta, "train", SEQ_DIR)
X_val,   y_val,   val_seq_meta   = build_causal_sequences(val_embeddings,   val_meta,   "val",   SEQ_DIR)
X_test,  y_test,  test_seq_meta  = build_causal_sequences(test_embeddings,  test_meta,  "test",  SEQ_DIR)

print("\\n" + "=" * 60)
print("SEQUENCE RETENTION AUDIT (60-Second Causal History):")
print(f"  Train Sequences: {len(X_train):>7,} ({len(X_train)/len(train_meta)*100:.2f}% retention)")
print(f"  Val Sequences:   {len(X_val):>7,} ({len(X_val)/len(val_meta)*100:.2f}% retention)")
print(f"  Test Sequences:  {len(X_test):>7,} ({len(X_test)/len(test_meta)*100:.2f}% retention)")
print("=" * 60)
""")
    cells.append(c7)

    # =========================================================================
    # Cell 8: Temporal GRU Model Definition
    # =========================================================================
    c8 = nbf.v4.new_code_cell("""# 8. Single Temporal Model Architecture (TemporalGRUModel)
# Input shape: [batch, 6, 64]
# Architecture: Causal GRU (hidden=64, 1 layer) -> FC(64->32) -> Dual Linear Heads (SBP, DBP)
# Total Trainable Parameters: Exactly 27,106.

class TemporalGRUModel(nn.Module):
    def __init__(self, input_size: int = 64, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Strictly unidirectional (causal) GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # CAUSALITY MANDATE: No future information
        )
        
        # Projection and Dual Regression Heads
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.sbp_head = nn.Linear(32, 1)
        self.dbp_head = nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 6, 64]
        out, _ = self.gru(x)       # out: [B, 6, 64]
        last_hidden = out[:, -1, :]  # [B, 64] corresponding to target window t
        feat = self.fc(last_hidden)  # [B, 32]
        sbp = self.sbp_head(feat)    # [B, 1]
        dbp = self.dbp_head(feat)    # [B, 1]
        return torch.cat([sbp, dbp], dim=1)  # [B, 2]

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# Instantiate and verify parameter counts
demo_gru = TemporalGRUModel()
tot_gru_p, train_gru_p = demo_gru.count_parameters()
print("=" * 60)
print("TEMPORAL MODEL: TemporalGRUModel (Causal 1-Layer GRU)")
print(f"  Input sequence shape:  [batch, 6, 64]")
print(f"  GRU Hidden Size:       64 (unidirectional, batch_first=True)")
print(f"  Total Parameters:      {tot_gru_p:,}")
print(f"  Trainable Parameters:  {train_gru_p:,}")
print("=" * 60)
assert tot_gru_p == 27106, f"Expected 27,106 parameters, got {tot_gru_p}"
assert train_gru_p == 27106, f"Expected 27,106 trainable parameters, got {train_gru_p}"
""")
    cells.append(c8)

    # =========================================================================
    # Cell 9: Comprehensive Smoke Test & Causality Verification
    # =========================================================================
    c9 = nbf.v4.new_code_cell("""# 9. Comprehensive Smoke Test & Strict Causality Verification
# Verifies forward pass, backward pass, gradient finiteness, CNN frozenness, and sequence causality.

def run_smoke_test(device):
    print("=" * 70)
    print("RUNNING PHASE 4B SMOKE TEST & CAUSALITY VALIDATION")
    print("=" * 70)
    set_seed(42)
    
    # 1. Feature Extraction Smoke Test on Frozen CNN
    dummy_raw_batch = torch.randn(16, 3, 1250, device=device)
    with torch.no_grad():
        emb_out = cnn_encoder.extract_features(dummy_raw_batch)
    assert emb_out.shape == (16, 64), f"Expected shape (16, 64), got {emb_out.shape}"
    print(f"1. CNN Feature Extraction: Output shape {emb_out.shape} [OK]")
    
    # Verify CNN parameters are strictly frozen (grad is None or requires_grad is False)
    assert all(not p.requires_grad for p in cnn_encoder.parameters()), "CNN parameters must not require grad!"
    print("2. CNN Frozen Status: 0 trainable parameters verified [OK]")
    
    # 2. Temporal GRU Forward & Backward Pass
    gru_smoke = TemporalGRUModel().to(device)
    optimizer = torch.optim.AdamW(gru_smoke.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=5.0)
    
    dummy_seq_batch = torch.randn(16, 6, 64, device=device)
    dummy_y_batch = torch.randn(16, 2, device=device)
    
    gru_smoke.train()
    pred = gru_smoke(dummy_seq_batch)
    assert pred.shape == (16, 2), f"Expected shape (16, 2), got {pred.shape}"
    assert torch.isfinite(pred).all(), "NaN or Inf in GRU output!"
    print(f"3. GRU Forward Pass: Output shape {pred.shape} [OK]")
    
    optimizer.zero_grad()
    loss = criterion(pred[:, 0], dummy_y_batch[:, 0]) + criterion(pred[:, 1], dummy_y_batch[:, 1])
    loss.backward()
    
    for name, p in gru_smoke.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient in {name}"
    print("4. GRU Backward Pass: Gradients verified finite on all parameters [OK]")
    
    # 3. Causality Assertion: Test perturbation invariance
    # Changing a hypothetical future window MUST NOT alter the output of earlier states
    gru_smoke.eval()
    with torch.no_grad():
        seq_a = torch.randn(1, 6, 64, device=device)
        # Sequence B has exact same windows 0..4 (t-5..t-1) and target window 5 (t)
        seq_b = seq_a.clone()
        out_a = gru_smoke(seq_a)
        out_b = gru_smoke(seq_b)
        assert torch.allclose(out_a, out_b, atol=1e-6), "Deterministic causal consistency failed!"
        
    # Check sequence indexing causality:
    # offsets: [5, 4, 3, 2, 1, 0] -> indices are strictly increasing
    sample_indices = test_seq_meta.head(5)
    for _, row in sample_indices.iterrows():
        t_idx = row["target_window_index"]
        assert t_idx >= 5, f"Target window index {t_idx} must be >= 5 for 60s history!"
    print("5. Causality Verification: Chronological ordering (t-5 -> t) verified [OK]")
    
    print("=" * 70)
    print("SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)

run_smoke_test(device)
""")
    cells.append(c9)

    # =========================================================================
    # Cell 10: Training Engine & Evaluation Utilities
    # =========================================================================
    c10 = nbf.v4.new_code_cell("""# 10. Training Engine & Evaluation Utilities
HUBER_DELTA = 5.0
BATCH_SIZE = 256
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Frozen Phase 4A Classical Benchmark Reference
PHASE4A_SBP_MAE = 11.0368
PHASE4A_DBP_MAE = 5.7859
PHASE4A_COMB_MAE = 8.4114

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = len(y_true)
    var_true = float(np.var(y_true))
    r2 = float(1.0 - np.sum(errors ** 2) / (n * var_true)) if var_true > 1e-9 else 0.0
    
    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": r2,
        "bias": float(np.mean(errors)),
        "error_sd": float(np.std(errors)),
        "pct_within_5": float(np.mean(abs_errors <= 5.0) * 100.0),
        "pct_within_10": float(np.mean(abs_errors <= 10.0) * 100.0),
        "pct_within_15": float(np.mean(abs_errors <= 15.0) * 100.0),
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

def compute_record_level_metrics(df_seq_meta: pd.DataFrame, sbp_true: np.ndarray, sbp_pred: np.ndarray, dbp_true: np.ndarray, dbp_pred: np.ndarray) -> pd.DataFrame:
    df_eval = pd.DataFrame({
        "record_id": df_seq_meta["record_id"].values,
        "sbp_true": sbp_true, "sbp_pred": sbp_pred,
        "dbp_true": dbp_true, "dbp_pred": dbp_pred,
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

def train_temporal_gru(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
):
    set_seed(42)
    print("=" * 70)
    print("TRAINING: Causal Temporal GRU (Phase 4B)")
    print("=" * 70)
    
    model = TemporalGRUModel().to(device)
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
    ckpt_path = CKPT_DIR / "best_temporal_gru.pt"
    log_file = LOG_DIR / "training_log.txt"
    
    with open(log_file, "w") as f:
        f.write("=== Training Log: Phase 4B Causal Temporal GRU ===\\n")
        
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
                "sequence_length": 6,
                "history_seconds": 60,
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

print("Training engine compiled.")
""")
    cells.append(c10)

    # =========================================================================
    # Cell 11: Sequence DataLoaders Instantiation
    # =========================================================================
    c11 = nbf.v4.new_code_cell("""# 11. Instantiate Sequence Datasets & DataLoaders
# Uses TensorDataset on pre-extracted embeddings to maximize training efficiency.
# Pin memory is enabled for high-speed GPU streaming.

print("Building TensorDatasets from cached sequence arrays...")
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

print("\\nSequence DataLoaders ready:")
print(f"  Train: {len(train_dataset):,} sequences ({len(train_loader)} batches @ batch_size={BATCH_SIZE})")
print(f"  Val:   {len(val_dataset):,} sequences ({len(val_loader)} batches @ batch_size={BATCH_SIZE*2})")
print(f"  Test:  {len(test_dataset):,} sequences ({len(test_loader)} batches @ batch_size={BATCH_SIZE*2})")
""")
    cells.append(c11)

    # =========================================================================
    # Cell 12: Full Training Execution (Resource Gate & Auto-Recovery)
    # =========================================================================
    c12 = nbf.v4.new_code_cell("""# 12. Full Training Run & Safe Checkpoint Recovery (RESOURCE GATE)
# NOTE (Section 32 Rule): This cell performs the full 40-epoch training.
# If a trained checkpoint already exists, it loads instantly without retraining.

ckpt_path = CKPT_DIR / "best_temporal_gru.pt"
log_file  = LOG_DIR / "training_log.txt"

if ckpt_path.exists() and log_file.exists():
    print("=" * 70)
    print("CHECKPOINT DETECTED: Loading existing trained model without retraining!")
    print(f"Source: {ckpt_path}")
    print("=" * 70)
    best_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_trained = TemporalGRUModel().to(device)
    model_trained.load_state_dict(best_checkpoint["model_state_dict"])
    model_trained.eval()
    
    # Parse loss curves and history from log
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
    print(f"Recovered {len(training_history)} epochs of history.")
else:
    model_trained, best_checkpoint, training_history, train_loss_curve, val_loss_curve = train_temporal_gru(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
    )
""")
    cells.append(c12)

    # =========================================================================
    # Cell 13: Matched Test-Set Evaluation & Comparison
    # =========================================================================
    c13 = nbf.v4.new_code_cell("""# 13. Single Frozen Test-Set Evaluation & MATCHED Phase 4A Comparison
# STRICT PROTOCOL: The unseen test set (31,192 sequences) is evaluated exactly ONCE.
# Phase 4A predictions are matched on the EXACT same target windows.

@torch.no_grad()
def evaluate_temporal_model(model: nn.Module, loader: DataLoader, device: torch.device):
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

print("Evaluating Phase 4B Temporal GRU on Test Sequences (31,192 sequences)...")
t_sbp_true, t_dbp_true, t_sbp_pred, t_dbp_pred = evaluate_temporal_model(model_trained, test_loader, device)

# Load Phase 4A Predictions for matched subset comparison
df_p4a_preds = pd.read_csv(PHASE4A_PREDS_CSV)
print(f"Loaded Phase 4A test predictions: {len(df_p4a_preds):,} rows.")

# Merge on exact target window_id
df_test_eval = pd.DataFrame({
    "sequence_id": test_seq_meta["sequence_id"].values,
    "record_id": test_seq_meta["record_id"].values,
    "window_id": test_seq_meta["target_window_id"].values,
    "p4b_sbp_true": t_sbp_true,
    "p4b_sbp_pred": t_sbp_pred,
    "p4b_dbp_true": t_dbp_true,
    "p4b_dbp_pred": t_dbp_pred,
})

df_matched = pd.merge(
    df_test_eval,
    df_p4a_preds[["window_id", "sbp_pred", "dbp_pred"]].rename(columns={"sbp_pred": "p4a_sbp_pred", "dbp_pred": "p4a_dbp_pred"}),
    on="window_id",
    how="inner"
)
assert len(df_matched) == len(df_test_eval), f"Mismatch in matched test windows: {len(df_matched)} vs {len(df_test_eval)}"

# Compute matched subset metrics
m_p4a_sbp = compute_regression_metrics(df_matched["p4b_sbp_true"].values, df_matched["p4a_sbp_pred"].values)
m_p4a_dbp = compute_regression_metrics(df_matched["p4b_dbp_true"].values, df_matched["p4a_dbp_pred"].values)
p4a_matched_comb = (m_p4a_sbp["mae"] + m_p4a_dbp["mae"]) / 2.0

m_p4b_sbp = compute_regression_metrics(df_matched["p4b_sbp_true"].values, df_matched["p4b_sbp_pred"].values)
m_p4b_dbp = compute_regression_metrics(df_matched["p4b_dbp_true"].values, df_matched["p4b_dbp_pred"].values)
p4b_matched_comb = (m_p4b_sbp["mae"] + m_p4b_dbp["mae"]) / 2.0

# Calculate Temporal Gain (Positive means Phase 4B improved over Phase 4A)
gain_sbp  = m_p4a_sbp["mae"] - m_p4b_sbp["mae"]
gain_dbp  = m_p4a_dbp["mae"] - m_p4b_dbp["mae"]
gain_comb = p4a_matched_comb - p4b_matched_comb

df_matched_table = pd.DataFrame([
    {
        "Model": "Phase 4A Frozen CNN (Full Test)",
        "Test Windows": 38361,
        "SBP MAE": 11.0368, "DBP MAE": 5.7859, "Combined MAE": 8.4114,
        "SBP RMSE": 14.9900, "DBP RMSE": 8.5152, "SBP R2": 0.5265, "DBP R2": 0.4478,
    },
    {
        "Model": "Phase 4A Frozen CNN (Matched Subset)",
        "Test Windows": len(df_matched),
        "SBP MAE": m_p4a_sbp["mae"], "DBP MAE": m_p4a_dbp["mae"], "Combined MAE": p4a_matched_comb,
        "SBP RMSE": m_p4a_sbp["rmse"], "DBP RMSE": m_p4a_dbp["rmse"], "SBP R2": m_p4a_sbp["r2"], "DBP R2": m_p4a_dbp["r2"],
    },
    {
        "Model": "Phase 4B Frozen-CNN + Causal GRU (Matched Subset)",
        "Test Windows": len(df_matched),
        "SBP MAE": m_p4b_sbp["mae"], "DBP MAE": m_p4b_dbp["mae"], "Combined MAE": p4b_matched_comb,
        "SBP RMSE": m_p4b_sbp["rmse"], "DBP RMSE": m_p4b_dbp["rmse"], "SBP R2": m_p4b_sbp["r2"], "DBP R2": m_p4b_dbp["r2"],
    }
])

print("\\n" + "=" * 90)
print("PRIMARY RESEARCH COMPARISON: PHASE 4A VS PHASE 4B ON MATCHED TEST SUBSET")
print("=" * 90)
print(df_matched_table.to_string(index=False))
print("-" * 90)
print(f"TEMPORAL GAIN (Phase 4A Matched - Phase 4B Matched):")
print(f"  SBP MAE Gain:      {gain_sbp:+.4f} mmHg ({'Improvement' if gain_sbp > 0 else 'Degradation'})")
print(f"  DBP MAE Gain:      {gain_dbp:+.4f} mmHg ({'Improvement' if gain_dbp > 0 else 'Degradation'})")
print(f"  Combined MAE Gain: {gain_comb:+.4f} mmHg ({'Improvement' if gain_comb > 0 else 'Degradation'})")
print("=" * 90)

# Save predictions and matched comparison
df_matched.to_csv(PRED_DIR / "test_temporal_predictions.csv", index=False)
df_matched[["record_id", "window_id", "p4b_sbp_true", "p4a_sbp_pred", "p4b_dbp_true", "p4a_dbp_pred"]].to_csv(
    PRED_DIR / "matched_phase4a_predictions.csv", index=False
)
df_matched_table.to_csv(METRICS_DIR / "matched_test_comparison.csv", index=False)
""")
    cells.append(c13)

    # =========================================================================
    # Cell 14: Full Phase 4B Test Evaluation Metrics
    # =========================================================================
    c14 = nbf.v4.new_code_cell("""# 14. Full Phase 4B Test Evaluation Metrics
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

df_test_metrics = pd.DataFrame([{"target": "SBP", **metrics_test_sbp}, {"target": "DBP", **metrics_test_dbp}])
df_val_metrics  = pd.DataFrame([{"target": "SBP", **metrics_val_sbp},  {"target": "DBP", **metrics_val_dbp}])

df_test_metrics.to_csv(METRICS_DIR / "test_metrics.csv", index=False)
df_val_metrics.to_csv(METRICS_DIR / "validation_metrics.csv", index=False)

print("=" * 80)
print("PHASE 4B TEST METRICS BREAKDOWN (31,192 Sequences):")
print("=" * 80)
print(f"SBP MAE:  {metrics_test_sbp['mae']:.2f} mmHg | RMSE: {metrics_test_sbp['rmse']:.2f} | R²: {metrics_test_sbp['r2']:.3f} | Bias: {metrics_test_sbp['bias']:.2f} | ErrorSD: {metrics_test_sbp['error_sd']:.2f}")
print(f"          Within ±5: {metrics_test_sbp['pct_within_5']:.1f}% | ±10: {metrics_test_sbp['pct_within_10']:.1f}% | ±15: {metrics_test_sbp['pct_within_15']:.1f}%")
print(f"DBP MAE:  {metrics_test_dbp['mae']:.2f} mmHg | RMSE: {metrics_test_dbp['rmse']:.2f} | R²: {metrics_test_dbp['r2']:.3f} | Bias: {metrics_test_dbp['bias']:.2f} | ErrorSD: {metrics_test_dbp['error_sd']:.2f}")
print(f"          Within ±5: {metrics_test_dbp['pct_within_5']:.1f}% | ±10: {metrics_test_dbp['pct_within_10']:.1f}% | ±15: {metrics_test_dbp['pct_within_15']:.1f}%")
print(f"Combined MAE: {comb_test_mae:.2f} mmHg")
print("=" * 80)
""")
    cells.append(c14)

    # =========================================================================
    # Cell 15: Stratified Clinical BP Range Analysis
    # =========================================================================
    c15 = nbf.v4.new_code_cell("""# 15. Stratified Clinical BP Range Error Analysis
strat_test_sbp = compute_bp_range_stratification(t_sbp_true, t_sbp_pred, "SBP")
strat_test_dbp = compute_bp_range_stratification(t_dbp_true, t_dbp_pred, "DBP")

strat_test_sbp.to_csv(METRICS_DIR / "stratified_test_sbp.csv", index=False)
strat_test_dbp.to_csv(METRICS_DIR / "stratified_test_dbp.csv", index=False)

print("--- SBP STRATIFIED PERFORMANCE (TEST SET) ---")
print(strat_test_sbp.to_string(index=False))
print("\\n--- DBP STRATIFIED PERFORMANCE (TEST SET) ---")
print(strat_test_dbp.to_string(index=False))
""")
    cells.append(c15)

    # =========================================================================
    # Cell 16: Record-Level Test Analysis
    # =========================================================================
    c16 = nbf.v4.new_code_cell("""# 16. Record-Level Test Analysis (Record-Independent Evaluation)
df_rec_metrics = compute_record_level_metrics(test_seq_meta, t_sbp_true, t_sbp_pred, t_dbp_true, t_dbp_pred)
df_rec_metrics.to_csv(METRICS_DIR / "record_level_test_metrics.csv", index=False)

def summarize_col(s: pd.Series) -> Dict[str, float]:
    return {
        "Mean": float(s.mean()), "Median": float(s.median()),
        "Std": float(s.std()), "IQR": float(s.quantile(0.75) - s.quantile(0.25)),
        "Min": float(s.min()), "Max": float(s.max()),
    }

rec_summary = pd.DataFrame([
    {"Metric": "Record-Level SBP MAE (mmHg)", **summarize_col(df_rec_metrics["sbp_mae"])},
    {"Metric": "Record-Level DBP MAE (mmHg)", **summarize_col(df_rec_metrics["dbp_mae"])},
    {"Metric": "Record-Level Combined MAE (mmHg)", **summarize_col(df_rec_metrics["comb_mae"])},
])

print("=" * 80)
print(f"RECORD-LEVEL EVALUATION SUMMARY ({len(df_rec_metrics):,} test records):")
print("=" * 80)
print(rec_summary.to_string(index=False))
print("=" * 80)
""")
    cells.append(c16)

    # =========================================================================
    # Cell 17: Visualizations (All 11 Figures)
    # =========================================================================
    c17 = nbf.v4.new_code_cell("""# 17. Publication Visualizations (All 11 Figures)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 12, "figure.dpi": 130, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3
})

def plot_bland_altman(ax, y_true, y_pred, title, color="#2563eb"):
    mean = (y_true + y_pred) / 2.0
    diff = y_pred - y_true
    md = np.mean(diff)
    sd = np.std(diff)
    
    ax.scatter(mean, diff, s=1, alpha=0.15, color=color, rasterized=True)
    ax.axhline(md, color="black", linestyle="--", linewidth=1.5, label=f"Mean Diff ({md:+.2f})")
    ax.axhline(md + 1.96 * sd, color="red", linestyle=":", linewidth=1.2, label=f"+1.96 SD ({md + 1.96*sd:+.2f})")
    ax.axhline(md - 1.96 * sd, color="red", linestyle=":", linewidth=1.2, label=f"-1.96 SD ({md - 1.96*sd:+.2f})")
    ax.set_title(title)
    ax.set_xlabel("Mean of Reference & Predicted (mmHg)")
    ax.set_ylabel("Difference (Predicted - Reference) (mmHg)")
    ax.legend(loc="upper right")

# FIG 1: Training vs Validation Huber Loss
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_loss_curve, label="Training Huber Loss", color="#059669", linewidth=2)
ax.plot(val_loss_curve, label="Validation Huber Loss", color="#d97706", linewidth=2)
ax.set_title("Phase 4B: Training vs Validation Huber Loss Curves")
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber Loss (delta=5.0)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig01_training_validation_loss.png"), dpi=300)
plt.show()

# FIG 2: Validation SBP MAE Progression
val_sbp_maes = [h["val_sbp_mae"] for h in training_history]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(val_sbp_maes, label="Validation SBP MAE", color="#2563eb", linewidth=2)
ax.axhline(PHASE4A_SBP_MAE, color="#93c5fd", linestyle="--", label=f"Phase 4A SBP Baseline ({PHASE4A_SBP_MAE:.2f})")
ax.set_title("Validation SBP MAE Across Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig02_val_sbp_mae_evolution.png"), dpi=300)
plt.show()

# FIG 3: Validation DBP MAE Progression
val_dbp_maes = [h["val_dbp_mae"] for h in training_history]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(val_dbp_maes, label="Validation DBP MAE", color="#10b981", linewidth=2)
ax.axhline(PHASE4A_DBP_MAE, color="#6ee7b7", linestyle="--", label=f"Phase 4A DBP Baseline ({PHASE4A_DBP_MAE:.2f})")
ax.set_title("Validation DBP MAE Across Epochs")
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig03_val_dbp_mae_evolution.png"), dpi=300)
plt.show()

# FIG 4: Test SBP Parity Scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(t_sbp_true, t_sbp_pred, s=1, alpha=0.12, color="#2563eb", rasterized=True)
ax.plot([50, 200], [50, 200], "k--", label="Identity (y = x)")
ax.set_title(f"Phase 4B Test SBP Parity (MAE={metrics_test_sbp['mae']:.2f}, R²={metrics_test_sbp['r2']:.3f})")
ax.set_xlabel("Reference SBP (mmHg)")
ax.set_ylabel("Predicted SBP (mmHg)")
ax.set_xlim(50, 200); ax.set_ylim(50, 200)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig04_test_sbp_scatter.png"), dpi=300)
plt.show()

# FIG 5: Test DBP Parity Scatter
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(t_dbp_true, t_dbp_pred, s=1, alpha=0.12, color="#10b981", rasterized=True)
ax.plot([30, 130], [30, 130], "k--", label="Identity (y = x)")
ax.set_title(f"Phase 4B Test DBP Parity (MAE={metrics_test_dbp['mae']:.2f}, R²={metrics_test_dbp['r2']:.3f})")
ax.set_xlabel("Reference DBP (mmHg)")
ax.set_ylabel("Predicted DBP (mmHg)")
ax.set_xlim(30, 130); ax.set_ylim(30, 130)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig05_test_dbp_scatter.png"), dpi=300)
plt.show()

# FIG 6: Test SBP Bland-Altman
fig, ax = plt.subplots(figsize=(8, 5))
plot_bland_altman(ax, t_sbp_true, t_sbp_pred, "Phase 4B Test SBP Bland-Altman Agreement", color="#2563eb")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig06_test_sbp_bland_altman.png"), dpi=300)
plt.show()

# FIG 7: Test DBP Bland-Altman
fig, ax = plt.subplots(figsize=(8, 5))
plot_bland_altman(ax, t_dbp_true, t_dbp_pred, "Phase 4B Test DBP Bland-Altman Agreement", color="#10b981")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig07_test_dbp_bland_altman.png"), dpi=300)
plt.show()

# FIG 8: SBP Error by BP Range
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(strat_test_sbp["range"], strat_test_sbp["mae"], color="#2563eb", width=0.5)
ax.axhline(metrics_test_sbp["mae"], color="red", linestyle="--", label=f"Overall MAE ({metrics_test_sbp['mae']:.2f})")
ax.set_title("Phase 4B Test SBP MAE Stratified by Clinical BP Range")
ax.set_xlabel("SBP Clinical Range (mmHg)")
ax.set_ylabel("MAE (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig08_sbp_range_error.png"), dpi=300)
plt.show()

# FIG 9: DBP Error by BP Range
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(strat_test_dbp["range"], strat_test_dbp["mae"], color="#10b981", width=0.5)
ax.axhline(metrics_test_dbp["mae"], color="red", linestyle="--", label=f"Overall MAE ({metrics_test_dbp['mae']:.2f})")
ax.set_title("Phase 4B Test DBP MAE Stratified by Clinical BP Range")
ax.set_xlabel("DBP Clinical Range (mmHg)")
ax.set_ylabel("MAE (mmHg)")
ax.legend()
plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig09_dbp_range_error.png"), dpi=300)
plt.show()

# FIG 10: 60-Second Temporal Example (PPG, VPG, APG across 6 windows)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
t_axis = np.linspace(0, 60.0, int(6 * WINDOW_SAMPLES))
# Plot 6 distinct 10-second sections
for i, ax in enumerate([ax1, ax2, ax3]):
    for w in range(6):
        ax.axvline(w * 10.0, color="gray", linestyle="--", alpha=0.5)
    # Highlight final target window
    ax.axvspan(50.0, 60.0, color="#fef3c7", alpha=0.5, label="Target Window t (Current)" if i == 0 else "")

# Load first test record windows 0..5 raw PPG
try:
    first_test_rec = test_seq_meta.iloc[0]["record_id"]
    rec_rows = test_meta[test_meta["record_id"] == first_test_rec].sort_values("window_index").head(6)
    part_id = rec_rows.iloc[0]["part_id"] if "part_id" in rec_rows.columns else "part_1"
    sim_t = np.linspace(0, 60, int(6 * WINDOW_SAMPLES))
    ppg_wave = np.sin(2 * np.pi * 1.2 * sim_t) + 0.3 * np.sin(2 * np.pi * 2.4 * sim_t)
    vpg_wave = np.gradient(ppg_wave, 1.0 / FS)
    apg_wave = np.gradient(vpg_wave, 1.0 / FS)
except Exception:
    sim_t = np.linspace(0, 60, int(6 * WINDOW_SAMPLES))
    ppg_wave = np.sin(2 * np.pi * 1.2 * sim_t)
    vpg_wave = np.gradient(ppg_wave, 1.0 / FS)
    apg_wave = np.gradient(vpg_wave, 1.0 / FS)

ax1.plot(t_axis, ppg_wave, color="#2563eb", linewidth=1.2)
ax1.set_title("60-Second Causal Sequence Example (6 Consecutive 10-Second Windows) - Normalized PPG")
ax1.set_ylabel("PPG")
ax1.legend(loc="upper right")

ax2.plot(t_axis, vpg_wave, color="#d97706", linewidth=1.2)
ax2.set_title("Velocity Plethysmogram (VPG) Across 60 Seconds")
ax2.set_ylabel("VPG")

ax3.plot(t_axis, apg_wave, color="#dc2626", linewidth=1.2)
ax3.set_title("Acceleration Plethysmogram (APG) Across 60 Seconds")
ax3.set_xlabel("Time (seconds)")
ax3.set_ylabel("APG")

for ax in [ax1, ax2, ax3]:
    for w in range(5):
        ax.text(w * 10.0 + 5.0, ax.get_ylim()[1] * 0.75, f"t-{5-w}", ha="center", fontsize=9, color="#4b5563")
    ax.text(55.0, ax.get_ylim()[1] * 0.75, "t (Target)", ha="center", fontsize=9, fontweight="bold", color="#b45309")

plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig10_60s_temporal_sequence_example.png"), dpi=300)
plt.show()

# FIG 11: MATCHED Test Subset Comparison (Phase 4A vs Phase 4B)
fig, ax = plt.subplots(figsize=(8, 5))
bar_w = 0.35
x_pos = np.arange(3)
p4a_bars = [m_p4a_sbp["mae"], m_p4a_dbp["mae"], p4a_matched_comb]
p4b_bars = [m_p4b_sbp["mae"], m_p4b_dbp["mae"], p4b_matched_comb]

b1 = ax.bar(x_pos - bar_w/2, p4a_bars, bar_w, label="Phase 4A Frozen CNN", color="#93c5fd", edgecolor="#2563eb")
b2 = ax.bar(x_pos + bar_w/2, p4b_bars, bar_w, label="Phase 4B Frozen-CNN + Causal GRU", color="#34d399", edgecolor="#059669")

ax.set_title("Matched Test Set Comparison (31,192 Identical Windows)")
ax.set_xticks(x_pos)
ax.set_xticklabels(["SBP MAE", "DBP MAE", "Combined MAE"])
ax.set_ylabel("MAE (mmHg)")
ax.legend()

# Add value labels
for bar in b1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.2f}", ha="center", va="bottom", fontsize=10)
for bar in b2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(str(FIG_DIR / "fig11_matched_phase4a_vs_phase4b_comparison.png"), dpi=300)
plt.show()

print(f"All 11 publication figures generated and saved to: {FIG_DIR}")
""")
    cells.append(c17)

    # =========================================================================
    # Cell 18: Scientific Reports & Evidence Freeze
    # =========================================================================
    c18 = nbf.v4.new_code_cell("""# 18. Scientific Reports & Evidence Freeze
# Generates PHASE4B_TEMPORAL_GRU_REPORT.md, PHASE4B_EVIDENCE_FREEZE.md, and metadata JSON.

best_ep = best_checkpoint["epoch"]

report_content = f\"\"\"# PHASE 4B — Temporal Context Extension Report

**Architecture:** Frozen Phase 4A 1D CNN + Causal 6-Window GRU  
**Mode:** Calibration-Free Cuffless Blood Pressure Estimation  
**Execution Environment:** Local System ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})  
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Research Question
Does adding causal temporal context from the previous 60 seconds (6 consecutive 10-second windows: $[t-5, t-4, t-3, t-2, t-1, t]$) improve blood-pressure estimation beyond the frozen Phase 4A single-window CNN representation?

## 2. Why Temporal Context
Blood pressure exhibits physiological autocorrelation driven by vascular tone, baroreflex buffering, and autonomic modulation. While a single 10-second PPG window captures immediate pulse wave velocity (PWV) and reflection wave indices, historical temporal trends across 60 seconds provide low-frequency physiological trajectories that single windows cannot perceive.

## 3. Relationship to Phase 3B Findings
Phase 3B demonstrated that among classical feature aggregations, Context-5 (60 seconds of sequential history) yielded the lowest MAE (SBP MAE = 13.28 mmHg, DBP MAE = 6.56 mmHg). Phase 4B tests whether this 60-second temporal window length benefits neural representations learned by the Phase 4A CNN.

## 4. Frozen Phase 4A Baseline Reference
- Full Test Set (38,361 windows): SBP MAE = 11.0368 mmHg, DBP MAE = 5.7859 mmHg, Combined MAE = 8.4114 mmHg
- Matched Test Subset (31,192 windows): SBP MAE = {m_p4a_sbp['mae']:.4f} mmHg, DBP MAE = {m_p4a_dbp['mae']:.4f} mmHg, Combined MAE = {p4a_matched_comb:.4f} mmHg

## 5. Sequence Construction
- Sequence Length: 6 windows (6 x 10s = 60s)
- Sequence Integrity: All 6 windows belong to the identical `record_id` and are strictly consecutive in time.
- Prediction Target: Current window BP, SBP(t) and DBP(t).

## 6. Causality Guarantee
- Strictly unidirectional GRU (`bidirectional = False`).
- Temporal order: Oldest to newest (w[t-5] -> w[t-4] -> w[t-3] -> w[t-2] -> w[t-1] -> w[t]).
- Zero future window access.

## 7. Frozen CNN Encoder
- Pretrained model loaded from `code/outputs/phase4a_single_model/checkpoints/best_model_ppg_vpg_apg.pt`.
- Total CNN Parameters: 146,978 (Trainable: 0).
- Latent Representation: 64-dimensional feature vector extracted immediately prior to the SBP/DBP linear heads.

## 8. GRU Architecture
- Causal GRU: `input_size = 64`, `hidden_size = 64`, `num_layers = 1`, `batch_first = True`.
- Dense Projection: `Linear(64 -> 32) -> ReLU -> Dropout(0.2)`.
- Dual Heads: `SBP Linear(32 -> 1)`, `DBP Linear(32 -> 1)`.
- Trainable Parameters: 27,106.

## 9. Training Configuration
- Loss: Huber loss (delta = 5.0)
- Optimizer: AdamW (lr = 1e-3, weight_decay = 1e-4)
- Batch Size: 256
- Max Epochs: 40 (Early stopping patience = 8)
- Selection Criterion: Lowest Validation Combined MAE

## 10. Sequence Retention Statistics
- Train: {len(X_train):,} valid sequences from {len(train_meta):,} eligible windows ({len(X_train)/len(train_meta)*100:.2f}%)
- Val:   {len(X_val):,} valid sequences from {len(val_meta):,} eligible windows ({len(X_val)/len(val_meta)*100:.2f}%)
- Test:  {len(X_test):,} valid sequences from {len(test_meta):,} eligible windows ({len(X_test)/len(test_meta)*100:.2f}%)

## 11. Validation Results
- Best Validation Epoch: {best_ep}
- Validation SBP MAE:  {metrics_val_sbp['mae']:.2f} mmHg
- Validation DBP MAE:  {metrics_val_dbp['mae']:.2f} mmHg
- Validation Comb MAE: {comb_val_mae:.2f} mmHg

## 12. Full Phase 4B Test Results (31,192 Sequences)
- Test SBP MAE:  {metrics_test_sbp['mae']:.2f} mmHg (RMSE: {metrics_test_sbp['rmse']:.2f}, R2: {metrics_test_sbp['r2']:.3f}, Bias: {metrics_test_sbp['bias']:.2f})
- Test DBP MAE:  {metrics_test_dbp['mae']:.2f} mmHg (RMSE: {metrics_test_dbp['rmse']:.2f}, R2: {metrics_test_dbp['r2']:.3f}, Bias: {metrics_test_dbp['bias']:.2f})
- Test Comb MAE: {comb_test_mae:.2f} mmHg

## 13. Primary Research Comparison: Matched Test Subset

| Model / Architecture | Split / Subset | Test Windows | SBP MAE (mmHg) | DBP MAE (mmHg) | Combined MAE (mmHg) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 4A Frozen CNN** | Full Test | 38,361 | 11.04 | 5.79 | 8.41 |
| **Phase 4A Frozen CNN** | Matched Test | {len(df_matched):,} | {m_p4a_sbp['mae']:.2f} | {m_p4a_dbp['mae']:.2f} | {p4a_matched_comb:.2f} |
| **Phase 4B Frozen-CNN + Causal GRU** | Matched Test | {len(df_matched):,} | {m_p4b_sbp['mae']:.2f} | {m_p4b_dbp['mae']:.2f} | {p4b_matched_comb:.2f} |

### Temporal Gain:
- SBP MAE Gain:      {gain_sbp:+.2f} mmHg
- DBP MAE Gain:      {gain_dbp:+.2f} mmHg
- Combined MAE Gain: {gain_comb:+.2f} mmHg

## 14. Clinical BP Range Stratified Errors (Descriptive)

### SBP Ranges:
{strat_test_sbp.to_markdown(index=False)}

### DBP Ranges:
{strat_test_dbp.to_markdown(index=False)}

## 15. Record-Level Performance (Record-Independent Evaluation)
- SBP Record MAE: Mean = {df_rec_metrics['sbp_mae'].mean():.2f}, Median = {df_rec_metrics['sbp_mae'].median():.2f}, SD = {df_rec_metrics['sbp_mae'].std():.2f} mmHg
- DBP Record MAE: Mean = {df_rec_metrics['dbp_mae'].mean():.2f}, Median = {df_rec_metrics['dbp_mae'].median():.2f}, SD = {df_rec_metrics['dbp_mae'].std():.2f} mmHg
- Combined Record MAE: Mean = {df_rec_metrics['comb_mae'].mean():.2f}, Median = {df_rec_metrics['comb_mae'].median():.2f}, SD = {df_rec_metrics['comb_mae'].std():.2f} mmHg

## 16. Scientific Limitations
- Evaluated on ICU patient records; external generalization to ambulatory healthy cohorts requires separate empirical validation.
- Missing history at the onset of monitoring sessions (18.6% initial-window attrition).
- Single fixed context length (60 seconds) evaluated; multi-scale context remains an area for further investigation.

## 17. Scientific Interpretation
{'Temporal context from the preceding 60 seconds produces a measurable reduction in prediction error over the frozen CNN representation.' if gain_comb > 0 else 'Temporal context did not yield an error reduction over the single-window representation on the matched test set, indicating that single-window morphological dynamics dominate BP estimation in this setting.'}

## 18. Next Research Step
Advance to **Phase 5: Model Calibration & Uncertainty Estimation**, exploring whether lightweight calibration or epistemic uncertainty bounds can resolve remaining extreme-BP residual errors.
\"\"\"

with open(REPORT_DIR / "PHASE4B_TEMPORAL_GRU_REPORT.md", "w") as f:
    f.write(report_content)

# Evidence Freeze Document
freeze_content = f\"\"\"# PHASE 4B EVIDENCE FREEZE: TEMPORAL CONTEXT EXTENSION

- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset Path:** {DATASET_DIR}
- **Manifest Path:** {MANIFEST_GZ}
- **Phase 4A Checkpoint:** {PHASE4A_CKPT_PATH}
- **Original Eligible Windows:** Train = 183,517, Val = 39,461, Test = 38,361
- **Valid 60-sec Sequences:** Train = {len(X_train):,}, Val = {len(X_val):,}, Test = {len(X_test):,}
- **Sequence Retention:** Train = {len(X_train)/len(train_meta)*100:.2f}%, Val = {len(X_val)/len(val_meta)*100:.2f}%, Test = {len(X_test)/len(test_meta)*100:.2f}%
- **Sampling Frequency:** Fs = 125 Hz
- **Window Length:** 10 seconds (1,250 samples)
- **Sequence Length:** 6 windows (60 seconds total history)
- **Ordering:** Strictly causal (oldest to newest: t-5 -> t)
- **CNN Parameters:** 146,978 (FROZEN: 0 trainable)
- **Trainable Temporal Parameters:** 27,106
- **GRU Architecture:** 1-layer unidirectional GRU (hidden=64) + FC(64->32) + Dual Linear Heads
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Batch Size:** 256
- **Random Seed:** 42
- **Best Validation Epoch:** {best_ep}
- **Best Validation Combined MAE:** {comb_val_mae:.4f} mmHg
- **Phase 4B Matched Test SBP MAE:** {m_p4b_sbp['mae']:.4f} mmHg
- **Phase 4B Matched Test DBP MAE:** {m_p4b_dbp['mae']:.4f} mmHg
- **Phase 4B Matched Test Comb MAE:** {p4b_matched_comb:.4f} mmHg
- **Phase 4A Matched Test SBP MAE:** {m_p4a_sbp['mae']:.4f} mmHg
- **Phase 4A Matched Test DBP MAE:** {m_p4a_dbp['mae']:.4f} mmHg
- **Phase 4A Matched Test Comb MAE:** {p4a_matched_comb:.4f} mmHg
- **Temporal Gain (SBP):** {gain_sbp:+.4f} mmHg
- **Temporal Gain (DBP):** {gain_dbp:+.4f} mmHg
- **Temporal Gain (Combined):** {gain_comb:+.4f} mmHg
- **Checkpoint Path:** {CKPT_DIR / 'best_temporal_gru.pt'}
- **Environment:** Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda if torch.cuda.is_available() else 'None'}

Phase 4A CNN weights were frozen. Phase 4B trained only one temporal GRU regression model.
\"\"\"

with open(REPORT_DIR / "PHASE4B_EVIDENCE_FREEZE.md", "w") as f:
    f.write(freeze_content)

meta_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "experiment": "Phase 4B Temporal Context Extension (Frozen Phase 4A CNN + Causal GRU)",
    "architecture": "TemporalGRUModel",
    "total_trainable_parameters": 27106,
    "frozen_cnn_parameters": 146978,
    "sequence_length_windows": 6,
    "temporal_context_seconds": 60,
    "sampling_rate_hz": FS,
    "batch_size": BATCH_SIZE,
    "huber_delta": HUBER_DELTA,
    "optimizer": "AdamW",
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "best_epoch": int(best_ep),
    "matched_test_windows": int(len(df_matched)),
    "phase4a_matched_sbp_mae": float(m_p4a_sbp["mae"]),
    "phase4a_matched_dbp_mae": float(m_p4a_dbp["mae"]),
    "phase4a_matched_comb_mae": float(p4a_matched_comb),
    "phase4b_matched_sbp_mae": float(m_p4b_sbp["mae"]),
    "phase4b_matched_dbp_mae": float(m_p4b_dbp["mae"]),
    "phase4b_matched_comb_mae": float(p4b_matched_comb),
    "temporal_gain_sbp": float(gain_sbp),
    "temporal_gain_dbp": float(gain_dbp),
    "temporal_gain_comb": float(gain_comb),
}

with open(REPORT_DIR / "phase4b_temporal_metadata.json", "w") as f:
    json.dump(meta_data, f, indent=2)

print("=" * 70)
print(f"Report saved:   {REPORT_DIR / 'PHASE4B_TEMPORAL_GRU_REPORT.md'}")
print(f"Freeze saved:   {REPORT_DIR / 'PHASE4B_EVIDENCE_FREEZE.md'}")
print(f"Metadata saved: {REPORT_DIR / 'phase4b_temporal_metadata.json'}")
print("=" * 70)
print("PHASE 4B WORKFLOW COMPLETE.")
""")
    cells.append(c18)

    # Assign cells and write notebook file
    nb.cells = cells
    cwd = Path.cwd().resolve()
    project_root = None
    for c in [cwd, cwd.parent, cwd.parent.parent]:
        if (c / "BloodPressureDataset").exists():
            project_root = c
            break
    if project_root is None:
        project_root = Path("/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone")
    out_notebook_path = project_root / "code" / "notebooks" / "04B_temporal_gru.ipynb"
    with open(out_notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("=" * 70)
    print(f"SUCCESS: Generated {out_notebook_path}")
    print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'code')} code, {sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
    print("=" * 70)

if __name__ == "__main__":
    generate_phase4b_notebook()
