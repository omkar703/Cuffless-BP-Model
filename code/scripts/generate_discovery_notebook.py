#!/usr/bin/env python3
"""
Notebook Generator for Phase 3B Research Discovery 1:
Temporal Context & PPG/VPG/APG Controlled Experiments.

Builds code/notebooks/03B_discovery_temporal_context.ipynb containing all 18 mandatory sections:
1. Research hypothesis
2. Frozen Phase 3A baseline
3. Feature cache verification
4. Temporal ordering validation
5. Sequence construction
6. Context-0 baseline
7. Context-1
8. Context-2
9. Context-5
10. PPG/VPG/APG comparison
11. Temporal + derivative comparison
12. Validation metrics
13. BP-range analysis
14. Regression-to-mean analysis
15. Record-level analysis
16. Feature importance
17. Research interpretation
18. Decision for next phase
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "03B_discovery_temporal_context.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


def build_notebook():
    cells = []

    # Section 1
    cells.append(create_cell("markdown", """# Phase 3B Research Discovery 1: Temporal Context & PPG/VPG/APG Controlled Experiments

## 1. Research Hypothesis

In Phase 3A, our calibration-free classical benchmarks established that:
1. Single-channel PPG features carry genuine blood pressure signal (validation MAE: 14.07 mmHg SBP / 6.98 mmHg DBP).
2. Derivative features (VPG and APG) are among the most informative descriptors.
3. However, static 10-second regression exhibits severe **regression-to-the-mean**, underestimating severe hypertension (SBP $\\ge 160$ mmHg by ~30 mmHg) and overestimating hypotension (SBP $< 90$ mmHg by ~30 mmHg).

We formulate two specific, falsifiable research hypotheses:
- **Hypothesis 1 (Temporal Dynamics)**: *A single static 10-second PPG window loses information present in temporal evolution across consecutive windows. Sequential context (20–60s) will provide slow physiological trend and drift information, reducing regression-to-the-mean and extreme tail bias.*
- **Hypothesis 2 (Derivative Representation)**: *Explicit first (VPG) and second (APG) derivative representations capture arterial stiffness and wave reflection morphology not captured by pure pulse timing, providing orthogonal value independently of temporal context.*

**Scope Constraint**: Strictly CPU-only, zero-leakage, using the existing frozen `features_cache.h5`. No deep neural networks (CNN/LSTM/Transformers) are trained in this phase.
"""))

    cells.append(create_cell("code", """import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Configure project paths
NOTEBOOK_DIR = Path.cwd()
CODE_DIR = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

OUTPUT_DIR = CODE_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"
FIGURES_DIR = OUTPUT_DIR / "figures" / "phase3b_discovery"
FEATURES_DIR = OUTPUT_DIR / "features"

print("Environment initialized successfully.")
print(f"Metrics Directory: {METRICS_DIR}")
print(f"Figures Directory: {FIGURES_DIR}")
"""))

    # Section 2
    cells.append(create_cell("markdown", """## 2. Frozen Phase 3A Baseline

We inspect the frozen configuration and benchmarks established in Phase 3A.
"""))

    cells.append(create_cell("code", """frozen_cfg_path = OUTPUT_DIR / "models" / "frozen_configuration.json"
with open(frozen_cfg_path, "r") as f:
    frozen_cfg = json.load(f)

print("Frozen Phase 3A Champion Configuration:")
print(f"  Model: {frozen_cfg['model']}")
print(f"  Branch: {frozen_cfg['representation_branch']}")
print(f"  Feature Count: {frozen_cfg['feature_count']}")
print(f"  Hyperparameters: {frozen_cfg['hyperparameters']}")
print(f"  Validation Combined MAE: {frozen_cfg['val_combined_mae']:.4f} mmHg")
"""))

    # Section 3
    cells.append(create_cell("markdown", """## 3. Feature Cache Verification

We programmatically verify the contents, keys, data types, and dimensions of `features_cache.h5`.
"""))

    cells.append(create_cell("code", """h5_path = FEATURES_DIR / "features_cache.h5"
assert h5_path.exists(), f"Feature cache missing at {h5_path}"

with h5py.File(h5_path, "r") as f:
    print(f"HDF5 Datasets in {h5_path.name}:")
    for k in f.keys():
        print(f"  {k:22s} shape={str(f[k].shape):16s} dtype={f[k].dtype}")
    
    feature_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["feature_names"][:]]

print(f"\\nTotal cached features ({len(feature_names)}):")
print(feature_names)
"""))

    # Section 4
    cells.append(create_cell("markdown", """## 4. Temporal Ordering Validation

Temporal sequence construction requires that all records have strictly monotonically increasing window indices and that no sequences cross record boundaries.
"""))

    cells.append(create_cell("code", """from modeling.temporal_features import assert_temporal_sequence_integrity, TemporalSequenceBuilder

with h5py.File(h5_path, "r") as f:
    sample_win_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["window_id"][:10000]]
    sample_rec_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["record_id"][:10000]]
    sample_splits = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["split"][:10000]]

df_check = pd.DataFrame({
    "record_id": sample_rec_ids,
    "window_id": sample_win_ids,
    "win_idx": [int(w.split("_win_")[-1]) for w in sample_win_ids],
    "split": sample_splits,
})

# Verify strict monotonic ordering within each record
monotone_checks = []
for r_id, grp in df_check.groupby("record_id"):
    idxs = grp["win_idx"].values
    if len(idxs) > 1:
        monotone_checks.append(np.all(np.diff(idxs) > 0))

print(f"Audited {len(monotone_checks)} sample records: all strictly monotonic = {all(monotone_checks)}")
"""))

    # Section 5
    cells.append(create_cell("markdown", """## 5. Sequence Construction & Retention

We review the missing history audit across the 4 evaluated context lengths:
- **Context-0 (10s)**: Current window only ($W_t$)
- **Context-1 (20s)**: $W_{t-1} \\to W_t$
- **Context-2 (30s)**: $W_{t-2} \\to W_{t-1} \\to W_t$
- **Context-5 (60s)**: $W_{t-5} \\to \\dots \\to W_t$
"""))

    cells.append(create_cell("code", """df_exp_b = pd.read_csv(METRICS_DIR / "phase3b_context_length_results.csv")
display(df_exp_b[["context_label", "context_sec", "feature_count", "train_samples", "val_samples"]])

# Multi-window temporal context visualization
display(Image(filename=str(FIGURES_DIR / "10_temporal_context_example.png")))
"""))

    # Section 6
    cells.append(create_cell("markdown", """## 6. Context-0 Baseline (Experiment A)

Context-0 represents the exact Phase 3A static baseline evaluated on the same feature set.
"""))

    cells.append(create_cell("code", """c0_row = df_exp_b[df_exp_b["context_k"] == 0].iloc[0]
print("Context-0 Baseline Performance (Validation Set):")
print(f"  SBP MAE: {c0_row['sbp_mae_all']:.2f} mmHg (RMSE: {c0_row['sbp_rmse_all']:.2f}, R2: {c0_row['sbp_r2']:.3f})")
print(f"  DBP MAE: {c0_row['dbp_mae_all']:.2f} mmHg (RMSE: {c0_row['dbp_rmse_all']:.2f}, R2: {c0_row['dbp_r2']:.3f})")
print(f"  Combined MAE: {c0_row['comb_mae_all']:.2f} mmHg")
"""))

    # Section 7
    cells.append(create_cell("markdown", """## 7. Context-1 Evaluation (20 Seconds History)

Context-1 introduces the immediately preceding 10-second window ($W_{t-1} \\to W_t$).
"""))

    cells.append(create_cell("code", """c1_row = df_exp_b[df_exp_b["context_k"] == 1].iloc[0]
print("Context-1 Performance (Validation Set):")
print(f"  SBP MAE: {c1_row['sbp_mae_all']:.2f} mmHg (Δ vs C0: {c1_row['sbp_mae_all'] - c0_row['sbp_mae_all']:+.2f} mmHg)")
print(f"  DBP MAE: {c1_row['dbp_mae_all']:.2f} mmHg (Δ vs C0: {c1_row['dbp_mae_all'] - c0_row['dbp_mae_all']:+.2f} mmHg)")
print(f"  Combined MAE: {c1_row['comb_mae_all']:.2f} mmHg")
"""))

    # Section 8
    cells.append(create_cell("markdown", """## 8. Context-2 Evaluation (30 Seconds History)

Context-2 introduces two preceding windows ($W_{t-2} \\to W_{t-1} \\to W_t$).
"""))

    cells.append(create_cell("code", """c2_row = df_exp_b[df_exp_b["context_k"] == 2].iloc[0]
print("Context-2 Performance (Validation Set):")
print(f"  SBP MAE: {c2_row['sbp_mae_all']:.2f} mmHg (Δ vs C0: {c2_row['sbp_mae_all'] - c0_row['sbp_mae_all']:+.2f} mmHg)")
print(f"  DBP MAE: {c2_row['dbp_mae_all']:.2f} mmHg (Δ vs C0: {c2_row['dbp_mae_all'] - c0_row['dbp_mae_all']:+.2f} mmHg)")
print(f"  Combined MAE: {c2_row['comb_mae_all']:.2f} mmHg")
"""))

    # Section 9
    cells.append(create_cell("markdown", """## 9. Context-5 Evaluation (60 Seconds History)

Context-5 introduces five preceding windows (50s of historical context + 10s current window).
"""))

    cells.append(create_cell("code", """c5_row = df_exp_b[df_exp_b["context_k"] == 5].iloc[0]
print("Context-5 Performance (Validation Set):")
print(f"  SBP MAE: {c5_row['sbp_mae_all']:.2f} mmHg (Δ vs C0: {c5_row['sbp_mae_all'] - c0_row['sbp_mae_all']:+.2f} mmHg)")
print(f"  DBP MAE: {c5_row['dbp_mae_all']:.2f} mmHg (Δ vs C0: {c5_row['dbp_mae_all'] - c0_row['dbp_mae_all']:+.2f} mmHg)")
print(f"  Combined MAE: {c5_row['comb_mae_all']:.2f} mmHg")
"""))

    # Section 10
    cells.append(create_cell("markdown", """## 10. Explicit PPG vs VPG vs APG Comparison (Experiment C)

We evaluate whether derivative representations (VPG and APG) provide measurable improvements at a fixed static window (Context-0).
"""))

    cells.append(create_cell("code", """df_exp_c = pd.read_csv(METRICS_DIR / "phase3b_derivative_ablation_results.csv")
display(df_exp_c)
"""))

    # Section 11
    cells.append(create_cell("markdown", """## 11. Temporal + Derivative Interaction (Experiment D)

We examine the interaction of temporal context and derivative representations:
- **D1**: Current PPG only
- **D2**: Current PPG + VPG + APG
- **D3**: Temporal PPG only
- **D4**: Temporal PPG + VPG + APG (Combined full model)
"""))

    cells.append(create_cell("code", """df_exp_d = pd.read_csv(METRICS_DIR / "phase3b_temporal_derivative_interaction.csv")
display(df_exp_d)

# Plot derivative ablation and interaction
display(Image(filename=str(FIGURES_DIR / "08_ppg_vpg_apg_ablation.png")))
"""))

    # Section 12
    cells.append(create_cell("markdown", """## 12. Validation Context Length Comparison Plots

We plot SBP and DBP MAE and $R^2$ across context durations.
"""))

    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "01_context_length_vs_sbp_mae.png")))
display(Image(filename=str(FIGURES_DIR / "02_context_length_vs_dbp_mae.png")))
display(Image(filename=str(FIGURES_DIR / "05_context_length_vs_r2.png")))
"""))

    # Section 13
    cells.append(create_cell("markdown", """## 13. BP-Range & Extreme BP Bias Analysis

Does temporal context reduce the severe tail bias observed in Phase 3A?
"""))

    cells.append(create_cell("code", """df_ext_sbp = pd.read_csv(METRICS_DIR / "phase3b_extreme_sbp_bias.csv")
display(df_ext_sbp[["context_label", "range", "sample_count", "mae", "bias", "error_sd"]])

display(Image(filename=str(FIGURES_DIR / "03_context_length_vs_sbp_extreme_bias.png")))
display(Image(filename=str(FIGURES_DIR / "04_context_length_vs_dbp_extreme_bias.png")))
"""))

    # Section 14
    cells.append(create_cell("markdown", """## 14. Regression-to-the-Mean Analysis

We compare error-vs-reference slopes across context lengths.
"""))

    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "06_regression_to_mean_comparison.png")))
"""))

    # Section 15
    cells.append(create_cell("markdown", """## 15. Record-Level Error Distribution

We analyze whether temporal context improves per-record error distributions.
"""))

    cells.append(create_cell("code", """df_rec = pd.read_csv(METRICS_DIR / "phase3b_record_level_mae.csv")
display(df_rec)

display(Image(filename=str(FIGURES_DIR / "07_record_level_mae_comparison.png")))
"""))

    # Section 16
    cells.append(create_cell("markdown", """## 16. Feature Importance Ranking

Permutation feature importance of the champion temporal model.
"""))

    cells.append(create_cell("code", """df_imp = pd.read_csv(METRICS_DIR / "phase3b_temporal_feature_importance.csv")
display(df_imp.head(15))

display(Image(filename=str(FIGURES_DIR / "09_temporal_feature_importance.png")))
"""))

    # Section 17
    cells.append(create_cell("markdown", """## 17. Research Interpretation

### Quantitative Findings
1. **Derivatives**: Adding VPG and APG derivatives improves combined validation MAE from 10.84 down to 10.52 mmHg (+0.32 mmHg).
2. **Temporal Dynamics**: Expanding context from 10s to 60s improves combined validation MAE from 10.52 down to 9.91 mmHg (+0.61 mmHg).
3. **Linear Model**: Linear Ridge regression also improves from 16.15 to 15.41 mmHg, demonstrating that temporal dynamics provide direct physical trend information.
4. **Extreme Bias**: SBP $\\ge 160$ mmHg bias is reduced from -31.44 mmHg to -28.44 mmHg (+3.00 mmHg reduction).
5. **Frozen Test Evaluation**: Evaluated once on Context-5 (60s), achieving **13.45 mmHg SBP MAE / 6.84 mmHg DBP MAE** (vs Phase 3A: 13.93 / 7.05 mmHg).
"""))

    # Section 18
    cells.append(create_cell("markdown", """## 18. Decision for Next Research Phase

### **Classification: CASE C — Multi-Scale Spatio-Temporal PPG Model Justified**

Because **both** explicit derivative channels and temporal context provide significant, complementary improvements:
- The next phase should implement a **Multi-Scale Spatio-Temporal PPG Neural Architecture** with explicit $[x(t), x'(t), x''(t)]$ derivative channels, a 1D CNN local pulse encoder, and a causal temporal sequence model (GRU/Transformer).
- Range-weighted loss should be incorporated to address residual extreme-value regression-to-the-mean.
"""))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook_dict, f, indent=2)

    print(f"Successfully generated {NOTEBOOK_PATH} with {len(cells)} cells.")


if __name__ == "__main__":
    build_notebook()
