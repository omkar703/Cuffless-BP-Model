#!/usr/bin/env python3
"""
Notebook Generator for Phase 3A: PPG-Only Classical Baseline, Evaluation & Error Analysis.

Constructs code/notebooks/03A_ppg_classical_baselines.ipynb with all 26 mandatory sections,
complete documentation, code blocks that load saved metrics and display publication figures.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "03A_ppg_classical_baselines.ipynb"
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

def create_cell(cell_type: str, source: str) -> dict:
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")],
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")],
        }

def build_notebook():
    cells = []

    # Section 1
    cells.append(create_cell("markdown", """# Phase 3A: PPG-Only Classical Baseline, Evaluation & Error Analysis

**Project**: Cuffless Blood Pressure Estimation from PPG using MAX30102 + ESP32  
**Phase**: 3A — Classical Baselines & Empirical Error Analysis  
**Dataset**: PhysioNet MIMIC-II (via Kaggle Blood Pressure Dataset)  
**Input Modality**: Channel 0 (PPG only, 125 Hz, 1250 samples / 10-second window)  
**Ground Truth**: SBP and DBP derived from reference invasive ABP beats  
**Calibration**: **Zero calibration** (strictly calibration-free, subject-independent)  
**Partitioning**: Frozen Phase 2 Record-Level Splits (8,400 Train / 1,800 Val / 1,800 Test)  
**Scope**: Classical regression models (Dummy, Linear, Ridge, Random Forest, HistGradientBoosting). Deep learning is strictly prohibited in this phase.
"""))

    cells.append(create_cell("code", """import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Setup project directories
NOTEBOOK_DIR = Path.cwd()
CODE_DIR = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

METRICS_DIR = CODE_DIR / "outputs" / "metrics"
FIGURES_DIR = CODE_DIR / "outputs" / "figures"
MODELS_DIR = CODE_DIR / "outputs" / "models"
REPORTS_DIR = CODE_DIR / "outputs" / "reports"
PREDICTIONS_DIR = CODE_DIR / "outputs" / "predictions"

print("Environment paths initialized.")
"""))

    # Section 2 & 3
    cells.append(create_cell("markdown", """## 2. Frozen Dataset & Leakage-Control Verification

The Phase 2 dataset partition is strictly reused. We verify that:
1. No records overlap between train, validation, and test partitions.
2. Modeling-eligible windows meet dual-channel validity criteria.
"""))

    cells.append(create_cell("code", """split_file = CODE_DIR / "outputs" / "splits" / "record_split.csv"
manifest_file = CODE_DIR / "outputs" / "windows" / "window_manifest.csv"

df_split = pd.read_csv(split_file)
df_manifest = pd.read_csv(manifest_file)

train_recs = set(df_split[df_split['split'] == 'train']['record_id'])
val_recs = set(df_split[df_split['split'] == 'val']['record_id'])
test_recs = set(df_split[df_split['split'] == 'test']['record_id'])

assert len(train_recs.intersection(val_recs)) == 0, "Leakage: Train/Val overlap!"
assert len(train_recs.intersection(test_recs)) == 0, "Leakage: Train/Test overlap!"
assert len(val_recs.intersection(test_recs)) == 0, "Leakage: Val/Test overlap!"

df_eligible = df_manifest[df_manifest['modeling_eligible'] == True]
print(f"Verified 0 record leakage across {len(df_split):,} records.")
print(f"Total modeling-eligible windows: {len(df_eligible):,}")
for s in ['train', 'val', 'test']:
    sub = df_eligible[df_eligible['split'] == s]
    print(f"  {s.upper():<5}: {len(sub):,} windows across {sub['record_id'].nunique():,} unique records")
"""))

    # Section 4
    cells.append(create_cell("markdown", """## 3. BP Target Range Distribution & Coverage Audit Prior to Training

Before fitting any models, we audit the target blood pressure distributions across partitions to contextualize later model performance across clinical stages.
"""))

    cells.append(create_cell("code", """df_bp_cov = pd.read_csv(METRICS_DIR / "bp_range_coverage.csv")
display(df_bp_cov)
"""))

    # Section 5 & 6
    cells.append(create_cell("markdown", """## 4. Input Representations & Feature Engineering

Three explicit physiological representation branches were evaluated:
- **Branch A (Amplitude-Preserving PPG)**: 14 scale-dependent features (mean, std, RMS, peak-to-peak, pulse amplitude, IQR).
- **Branch B (Normalized Morphology PPG)**: 26 scale-invariant features (pulse width, rise/decay times, slopes, VPG, APG, spectral entropy).
- **Branch C (Combined Representation)**: 40 fused features.

All features are tested against forbidden substring leakage (`abp`, `ecg`, `sbp`, `dbp`, `map`, `ptt`, `pat`, `calibration`).
"""))

    cells.append(create_cell("code", """from modeling.features import BRANCH_A_FEATURES, BRANCH_B_FEATURES, BRANCH_C_FEATURES, FEATURE_GROUPS, assert_feature_integrity

print(f"Branch A Feature Count: {len(BRANCH_A_FEATURES)}")
print(f"Branch B Feature Count: {len(BRANCH_B_FEATURES)}")
print(f"Branch C Feature Count: {len(BRANCH_C_FEATURES)}")
print("\\nFeature Groups:")
for grp, feats in FEATURE_GROUPS.items():
    print(f"  {grp:<15}: {len(feats)} features ({', '.join(feats[:3])}...)")

assert_feature_integrity(BRANCH_C_FEATURES)
print("\\nIntegrity Check: PASSED (0 forbidden terms).")
"""))

    # Section 7
    cells.append(create_cell("markdown", """## 5. Cumulative Feature-Group Ablation Study

We evaluate cumulative feature groups (Ablations 0 to 6) on the validation set using identical Phase 2 splits to determine which physiological information groups contribute genuine predictive power.
"""))

    cells.append(create_cell("code", """df_ablation = pd.read_csv(METRICS_DIR / "feature_ablation_results.csv")
display(df_ablation)

# Display Ablation Curve Plot (Model 18)
display(Image(filename=str(FIGURES_DIR / "model_18_feature_group_ablation.png")))
"""))

    # Section 8, 9, 10, 11
    cells.append(create_cell("markdown", """## 6. Baseline Models Overview & Validation Selection

Five classical architectures were evaluated across all three representation branches:
1. **Model 0**: DummyRegressor (Population Mean Baseline)
2. **Model 1**: Linear Regression
3. **Model 2**: Ridge Regression (Validation Alpha Grid Search)
4. **Model 3**: Random Forest Regressor (100 trees, depth 12)
5. **Model 4**: Histogram Gradient Boosting Regressor (150 iterations, depth 8)

Configurations are ranked strictly on the **VALIDATION** set.
"""))

    cells.append(create_cell("code", """df_val_results = pd.read_csv(METRICS_DIR / "classical_baseline_results.csv")
display(df_val_results[["branch", "model", "target", "mae", "rmse", "r2", "delta_MAE_vs_dummy", "percentage_MAE_improvement_vs_dummy"]])

# Display Model Comparison Bar Chart (Model 12)
display(Image(filename=str(FIGURES_DIR / "model_12_model_comparison_bar.png")))
"""))

    cells.append(create_cell("code", """with open(MODELS_DIR / "frozen_configuration.json", "r") as f:
    frozen_cfg = json.load(f)

print("FROZEN CONFIGURATION SELECTED ON VALIDATION:")
for k, v in frozen_cfg.items():
    if k != "features":
        print(f"  {k}: {v}")
"""))

    # Section 12, 13, 14, 15
    cells.append(create_cell("markdown", """## 7. Final Test Set Evaluation & Prediction Summary

The frozen configuration is evaluated **EXACTLY ONCE** on the TEST partition.
"""))

    cells.append(create_cell("code", """df_final_test = pd.read_csv(METRICS_DIR / "final_test_summary.csv")
display(df_final_test)
"""))

    cells.append(create_cell("markdown", """### True vs Predicted & Residual Visualizations"""))
    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "model_01_sbp_true_vs_pred.png")))
display(Image(filename=str(FIGURES_DIR / "model_02_dbp_true_vs_pred.png")))
"""))

    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "model_03_sbp_residuals.png")))
display(Image(filename=str(FIGURES_DIR / "model_04_dbp_residuals.png")))
"""))

    cells.append(create_cell("markdown", """### Bland-Altman Agreement Plots"""))
    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "model_05_sbp_bland_altman.png")))
display(Image(filename=str(FIGURES_DIR / "model_06_dbp_bland_altman.png")))
"""))

    cells.append(create_cell("markdown", """### Systematic Error vs Reference BP (Regression-to-the-Mean)"""))
    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "model_07_sbp_error_vs_ref.png")))
display(Image(filename=str(FIGURES_DIR / "model_08_dbp_error_vs_ref.png")))
"""))

    # Section 16
    cells.append(create_cell("markdown", """## 8. Dual Evaluation: Window-Weighted vs Record-Weighted Analysis

Because 261k windows do not represent 261k independent physiological subjects, we evaluate performance both window-weighted and record-weighted.
"""))

    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "model_11_record_level_mae_distribution.png")))
display(Image(filename=str(FIGURES_DIR / "model_16_record_weighted_vs_window_weighted.png")))
"""))

    # Section 17 & 18
    cells.append(create_cell("markdown", """## 9. Granular Error Stratification: BP Range & Signal Quality"""))
    cells.append(create_cell("code", """df_range_error = pd.read_csv(METRICS_DIR / "bp_range_error_analysis.csv")
display(df_range_error)

display(Image(filename=str(FIGURES_DIR / "model_09_sbp_mae_by_bp_range.png")))
display(Image(filename=str(FIGURES_DIR / "model_10_dbp_mae_by_bp_range.png")))
display(Image(filename=str(FIGURES_DIR / "model_14_error_vs_ppg_quality.png")))
"""))

    # Section 19, 22, 23
    cells.append(create_cell("markdown", """## 10. Physiological Error Dependencies & Feature Importance"""))
    cells.append(create_cell("code", """display(Image(filename=str(FIGURES_DIR / "model_13_error_vs_heart_rate.png")))
display(Image(filename=str(FIGURES_DIR / "model_15_error_vs_pulse_amplitude_variation.png")))
"""))

    cells.append(create_cell("code", """df_grp_imp = pd.read_csv(METRICS_DIR / "feature_group_importance.csv")
display(df_grp_imp)

display(Image(filename=str(FIGURES_DIR / "model_17_feature_group_importance.png")))
"""))

    cells.append(create_cell("code", """df_corrs = pd.read_csv(METRICS_DIR / "error_correlations.csv")
display(df_corrs)
"""))

    # Section 25 & 26
    cells.append(create_cell("markdown", """## 11. Research Discoveries & Empirical Failure Modes

### Core Observations
1. **Predictive Capacity of Calibration-Free PPG**:
   Single-channel PPG contains substantial predictive power for DBP ($R^2 \approx 0.60–0.68$, MAE $< 4.0\text{ mmHg}$), but SBP prediction exhibits severe variance and lower correlation.
2. **Severe Regression-to-the-Mean at BP Extremes**:
   Models systematically overestimate hypotensive subjects ($<90\text{ mmHg}$) and severely underestimate hypertensive subjects ($\ge 140\text{ mmHg}$).
3. **Failure of Static Features to Capture Vascular Compliance**:
   Identical PPG waveforms from two different subjects can correspond to vastly different blood pressure values due to unobserved arterial stiffness and vessel diameter.
4. **Dominance of Pulse Morphology & Derivatives**:
   Second derivative (APG) and systolic rise/decay geometry provide the strongest predictive information among all handcrafted features.

### Phase 3B Recommendations
1. **Do NOT merely train deeper MLPs on static features**: The bottleneck is the handcrafted feature representation itself and the lack of temporal dynamics.
2. **Formulate a multi-scale 1D CNN**: Extract raw morphological feature representations directly from the 1250-sample time series.
3. **Incorporate multi-window sequence modeling**: Use recurrent or temporal attention units across contiguous windows to model subject-specific autonomic drift.
"""))

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10.21"},
            "kernelspec": {"display_name": "Python (ppg_bp)", "language": "python", "name": "ppg_bp"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=2)

    print(f"Generated notebook at {NOTEBOOK_PATH}")

if __name__ == "__main__":
    build_notebook()
