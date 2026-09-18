# Phase 1: PPG-Only Blood Pressure Estimation — Dataset Cleaning, Validation & Visualization

## Overview
This package implements Phase 1 of the B.Tech Capstone Project for **cuffless blood pressure estimation using Photoplethysmography (PPG) alone**.

The primary goal of Phase 1 is to audit, clean, validate, and statistically characterize the PhysioNet MIMIC-II / Kaggle Blood Pressure Dataset (`BloodPressureDataset`) while preserving original files and preparing the exact data foundation needed for subsequent ML/DL modeling and MAX30102 hardware integration.

---

## Directory Structure
```
code/
├── environment.yml                  # Reproducible Conda environment configuration (ppg_bp)
├── README.md                        # Documentation and usage guide
├── config/
│   └── config.py                    # Centralized hyperparameters, filter settings, physiological bounds
├── data/
│   ├── __init__.py
│   ├── loader.py                    # Memory-efficient record generator from MAT files
│   ├── quality_control.py           # Multi-criteria signal validation and anomaly detection
│   ├── preprocessing.py             # Zero-phase offline research bandpass filter and ABP beat extractor
│   └── dataset_statistics.py        # Aggregate metrics, durations, and BP distributions
├── visualization/
│   ├── __init__.py
│   ├── signal_plots.py              # Publication-quality waveform plots (raw vs filtered PPG, ABP)
│   └── dataset_plots.py             # Dataset-level distributions (duration, amplitudes, SBP, DBP, rejections)
├── utils/
│   ├── __init__.py
│   └── logging_utils.py             # Formatted terminal and file logging
├── notebooks/
│   └── 01_dataset_cleaning_and_visualization.ipynb  # Interactive 14-section research notebook
├── scripts/
│   └── run_dataset_analysis.py      # Standalone CLI runner script
└── outputs/
    ├── figures/                     # Generated visual plots (.png)
    ├── statistics/                  # Manifest CSV, summary JSON, and audit report
    └── cleaned/                     # Reserved for derived metadata/caches
```

---

## Getting Started

### 1. Activate Environment
```bash
conda activate ppg_bp
```

### 2. Run CLI Dataset Analysis
```bash
python scripts/run_dataset_analysis.py --full-run
```
Or for a rapid test run across a subset:
```bash
python scripts/run_dataset_analysis.py --sample-parts 2
```

### 3. Open the Interactive Notebook
```bash
jupyter notebook notebooks/01_dataset_cleaning_and_visualization.ipynb
```
Select kernel: **PPG BP**.
