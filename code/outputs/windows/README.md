# Phase 2 Windows Dataset Architecture & Manifest Documentation

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
