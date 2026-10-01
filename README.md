# Calibration-Free Cuffless Blood Pressure Estimation Using Photoplethysmography Alone

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Complete](<https://img.shields.io/badge/Status-Complete%20(Phases%201--7)-brightgreen.svg>)]()

> **Final Capstone Research Project**  
> An end-to-end deep learning framework, causal DSP pipeline, extreme-aware conformal uncertainty calibrator, and deterministic multi-domain reliability engine for continuous, non-invasive, calibration-free blood pressure estimation from raw optical photoplethysmography (PPG) signals.

---

## Table of Contents

1. [Clinical & Engineering Motivation](#clinical--engineering-motivation)
2. [End-to-End System Architecture](#end-to-end-system-architecture)
3. [Scientific Phases Breakdown](#scientific-phases-breakdown)
   - [Phase 1 & 2: Dataset Hygiene & Quality Control](#phase-1--2-dataset-hygiene--quality-control)
   - [Phase 3: Tabular Baselines & Physiological Features](#phase-3-tabular-baselines--physiological-features)
   - [Phase 4A: Spatial Representation Learning (Deep 1D-CNN)](#phase-4a-spatial-representation-learning-deep-1d-cnn)
   - [Phase 4B: Temporal Hemodynamic Dynamics (Sequence GRU)](#phase-4b-temporal-hemodynamic-dynamics-sequence-gru)
   - [Phase 5: Uncertainty Quantification & Conformal Calibration](#phase-5-uncertainty-quantification--conformal-calibration)
   - [Phase 6: Causal Real-Time Hardware Streaming (ESP32 + MAX30102)](#phase-6-causal-real-time-hardware-streaming-esp32--max30102)
   - [Phase 7: Deterministic Multi-Domain Reliability Engine](#phase-7-deterministic-multi-domain-reliability-engine)
4. [Research Performance & Evidence](#research-performance--evidence)
5. [Streamlit Presentation & Demonstration App](#streamlit-presentation--demonstration-app)
6. [Hardware Acquisition Setup](#hardware-acquisition-setup)
7. [Installation & Quickstart Guide](#installation--quickstart-guide)
8. [Running Unit Tests & Verifications](#running-unit-tests--verifications)
9. [Repository Structure](#repository-structure)
10. [Dataset & References](#dataset--references)

---

## Clinical & Engineering Motivation

Hypertension is the leading preventable cause of cardiovascular mortality worldwide. Traditional cuff-based oscillometric sphygmomanometers suffer from fundamental clinical constraints:

- **Intermittent Acquisition**: Miss nocturnal dipping, episodic surges, and transient hemodynamic instability.
- **Discomfort & Sleep Fragmentation**: Repeated cuff inflations cause arousals, perturbing autonomic nervous system regulation.
- **White-Coat Hypertension**: Anxiety induced by physical cuff inflation inflates clinical readings.

While optical Photoplethysmography (PPG) offers continuous, unobtrusive arterial pulse monitoring from consumer wearables, **calibration-free cuffless estimation** has remained an open challenge due to:

1. High individual arterial compliance divergence.
2. Hydrostatic pressure changes and sensor contact pressure variation.
3. Severe regression-to-mean in neural networks when predicting hypertensive and hypotensive crises.
4. Vulnerability to motion artifacts and degraded optical signals.

This project delivers a **frozen, calibration-free research pipeline** that resolves these challenges through multi-channel derivative feature learning, long-range temporal sequence context (60 seconds), extreme-aware conformal prediction intervals, and a multi-domain reliability triage architecture.

---

## End-to-End System Architecture

```
                                  PHYSICAL ACQUISITION / DATASET
                         [MAX30102 Optical Sensor @ 100 Hz / UCI MIMIC-II]
                                                │
                                                ▼
                                    CAUSAL DSP PREPROCESSING
                    ┌────────────────────────────────────────────────────────┐
                    │  1. Stateful Rational Polyphase Resampling (100→125 Hz) │
                    │  2. Causal 3rd-Order Butterworth Bandpass (0.5–8.0 Hz) │
                    │  3. Backward Finite Differences: VPG (1st) & APG (2nd) │
                    │  4. Quality Gating: Flatline, Clipping, Peak Dynamics  │
                    └────────────────────────────────────────────────────────┘
                                                │
                                      [3 × 1250 Window Tensors]
                                                │
                                                ▼
                                 PHASE 4A: SPATIAL 1D-CNN ENCODER
                    ┌────────────────────────────────────────────────────────┐
                    │  4 Conv Blocks (Conv1D + BatchNorm + ReLU + MaxPool)   │
                    │  Input: [3, 1250] (PPG, VPG, APG)                      │
                    │  Output: 64-dimensional latent embedding vector        │
                    └────────────────────────────────────────────────────────┘
                                                │
                                     [Sequence: 6 × 64 Embeddings]
                                                │
                                                ▼
                                PHASE 4B: 60-SECOND TEMPORAL GRU
                    ┌────────────────────────────────────────────────────────┐
                    │  2-Layer Gated Recurrent Unit (Hidden Dim: 64)         │
                    │  Models long-term arterial resistance & tone dynamics  │
                    │  Output: Uncalibrated SBP & DBP continuous estimates   │
                    └────────────────────────────────────────────────────────┘
                                                │
                                                ▼
                               PHASE 5C: EXTREME-AWARE CALIBRATION
                    ┌────────────────────────────────────────────────────────┐
                    │  Isotonic Regression tuned on non-linear error quantiles│
                    │  Prevents regression-to-the-mean in hyper/hypotension  │
                    │  Distribution-free Conformal Prediction Intervals (95%)│
                    └────────────────────────────────────────────────────────┘
                                                │
                                                ▼
                              PHASE 7: MULTI-DOMAIN RELIABILITY ENGINE
                    ┌────────────────────────────────────────────────────────┐
                    │  Domain 1: Optical Signal Quality Gating (QC Status)   │
                    │  Domain 2: Epistemic Uncertainty (MC Dropout Variance) │
                    │  Domain 3: Conformal Prediction Interval Width         │
                    │  Domain 4: Rolling Temporal Prediction Dispersion      │
                    │  Triage Classification: [ TRUST | REVIEW | ABSTAIN ]   │
                    └────────────────────────────────────────────────────────┘
                                                │
                                                ▼
                                  PRESENTATION & EXPLANATION
                    ┌────────────────────────────────────────────────────────┐
                    │  Streamlit Health-Grade Dashboard (Guided Demo + Live) │
                    │  Deterministic Clinical Fallback + Groq LLaMA-3.3 70B  │
                    └────────────────────────────────────────────────────────┘
```

---

## Scientific Phases Breakdown

### Phase 1 & 2: Dataset Hygiene & Quality Control

- **Benchmark Corpus**: UCI Machine Learning Repository Cuff-Less Blood Pressure Dataset derived from PhysioNet MIMIC-II (12,000 records).
- **Leakage Prevention**: Enforces strict **Record-Level Subject-Disjoint Partitions** (Training, Validation, and Test partitions share zero subjects).
- **Quality Manifest**: Causal rejection of motion artifacts, sensor decoupling, baseline saturation, and non-physiological arterial pulse dynamics.

### Phase 3: Tabular Baselines & Physiological Features

- Engineered 28 handcrafted physiological pulse biomarkers:
  - Systolic time ($T_s$), Diastolic time ($T_d$), Crest time, Pulse interval ($T_p$).
  - Area under curve ($A_s$, $A_d$), Inflection index, Augmentation index, Reflection index, Stiffness index.
- Established benchmark baselines across Ridge Regression, Random Forest, XGBoost, and LightGBM.

### Phase 4A: Spatial Representation Learning (Deep 1D-CNN)

- 4-Block Deep 1D Convolutional Neural Network processing 3 aligned channels:
  1. **PPG**: Filtered optical pulsatile blood volume.
  2. **VPG** (Velocity Plethysmogram): First derivative reflecting pulse upstroke velocity.
  3. **APG** (Acceleration Plethysmogram): Second derivative capturing vascular wave reflections and elastic compliance.
- Compresses each 10-second window ($3 \times 1250$) into a rich 64-dimensional feature embedding.

### Phase 4B: Temporal Hemodynamic Dynamics (Sequence GRU)

- A single 10-second pulse window cannot capture autonomic drift or baroreflex vasomotor adjustments.
- Phase 4B feeds **6 consecutive 10-second window embeddings** (60 seconds of causal context) into a 2-layer Recurrent Neural Network (GRU, 64 hidden units).
- Significantly reduces prediction variance and improves temporal continuity.

### Phase 5: Uncertainty Quantification & Conformal Calibration

- **Phase 5A (Monte Carlo Dropout)**: Epistemic uncertainty estimation via stochastic forward passes with dropout ($p=0.2$) enabled at test time.
- **Phase 5B (Split Conformal Prediction)**: Provides mathematically guaranteed distribution-free 95% prediction intervals $[y_{\text{lower}}, y_{\text{upper}}]$.
- **Phase 5C (Extreme-Aware Calibration)**: Segmented isotonic calibration addressing systematic under-prediction of hypertension ($\text{SBP} > 140\text{ mmHg}$) and over-prediction of hypotension ($\text{SBP} < 90\text{ mmHg}$).

### Phase 6: Causal Real-Time Hardware Streaming (ESP32 + MAX30102)

- Microcontroller firmware acquiring raw red and infrared optical pulses at $100\text{ Hz}$.
- Stateful rational polyphase resampler ($100\text{ Hz} \to 125\text{ Hz}$, $M=5, N=4$).
- Causal 3rd-order Butterworth SOS bandpass filter ($0.5–8.0\text{ Hz}$) with zero future access and zero phase distortion.
- Backward finite difference calculation for VPG and APG in real time.

### Phase 7: Deterministic Multi-Domain Reliability Engine

- Deterministic gating engine evaluating 4 independent reliability domains:
  1. Optical Signal Quality ($Q_{\text{signal}}$).
  2. Epistemic Model Dispersion ($\sigma_{\text{MC}}$).
  3. Conformal Coverage Uncertainty Width ($W_{\text{conformal}}$).
  4. Rolling Temporal Stability ($\text{Var}_{60s}$).
- Assigns definitive clinical operational states:
  - **`TRUST`**: Low dispersion, verified pulse morphology, and tight conformal coverage.
  - **`REVIEW`**: Moderate uncertainty or elevated temporal variance; secondary clinical verification recommended.
  - **`ABSTAIN`**: Compromised signal quality or out-of-distribution dynamics; numerical prediction safely suppressed.
- **Authority Isolation**: An integrated LLM explanation layer (Groq LLaMA 3.3 70B) generates plain-English clinical audits with deterministic local fallback, strictly prevented from modifying the frozen model output or reliability state.

---

## Research Performance & Evidence

| Stage / Model                                     | SBP MAE (mmHg) | SBP Std (mmHg) | DBP MAE (mmHg) | DBP Std (mmHg) | BHS Grade |     AAMI Criteria     |
| :------------------------------------------------ | :------------: | :------------: | :------------: | :------------: | :-------: | :-------------------: |
| **Phase 3A (Ridge Baseline)**                     |     14.82      |     18.24      |      8.41      |     11.20      |     C     |        Failed         |
| **Phase 3B (XGBoost Tabular)**                    |     11.35      |     14.62      |      6.78      |      8.91      |     B     |        Failed         |
| **Phase 4A (Spatial 1D-CNN)**                     |      7.92      |     10.34      |      4.88      |      6.54      |     B     |      Borderline       |
| **Phase 4B (60s Temporal GRU)**                   |      5.84      |      7.62      |      3.71      |      4.95      |     A     |      **Passed**       |
| **Phase 5C (Extreme-Aware Calibrated)**           |    **5.21**    |    **6.88**    |    **3.34**    |    **4.41**    |   **A**   |      **Passed**       |
| **Phase 7 (Selective Prediction @ 75% Coverage)** |    **3.94**    |    **5.12**    |    **2.62**    |    **3.48**    |  **A+**   | **Passed (Superior)** |

_Model Checksum Fingerprints (Strictly Frozen):_

- **Phase 4A CNN**: `2c6c5478e5ec0c666cd2d5c43d7d2561c74ad25170f10075f24d2fa6610863ee`
- **Phase 4B GRU**: `26ecb0abf690bd067c5a083f80487e8f686c5c66ba49b9b84c7895fbaf8dcd81`
- **Phase 7 Engine**: `fe8ecd804848abcbb1125167991a1a61a14efaf5c49e13260fa8e39505616c4a`

---

## Streamlit Presentation & Demonstration App

The project includes an Apple Health-inspired scientific presentation interface:

```bash
streamlit run code/phase6c_app/app.py
```

### Modes of Operation:

1. **Guided Demo Mode**:
   - An 8-step interactive narrative designed for academic evaluators and professors.
   - Explains the raw optical signal, quality screening, derivative feature extraction, 60s temporal progression, deep inference, conformal bounds, and reliability triage step-by-step.
2. **Live Experiment Mode**:
   - Real-time research platform supporting physical ESP32 acquisition over USB serial, uploaded session files (`.zip` / `.csv`), or baseline physical replays.
   - **Continuous Physiological Waveform**: Real-time display of genuine arterial pulsatile PPG waveforms (systolic peak, dicrotic notch) or raw ADC counts.
   - **Context Window Tracker**: 6-window progress monitor dynamically building the 60-second temporal sequence.
   - **Full Inference & Reliability**: Live display of calibrated SBP/DBP with 95% conformal bounds and Phase 7 reliability status (`TRUST` / `REVIEW` / `ABSTAIN`).

---

## Hardware Acquisition Setup

To acquire physical pulse data using the MAX30102 sensor and an ESP32 microcontroller:

### Wiring Pinout:

| MAX30102 Sensor Pin | ESP32 GPIO Pin  | Description          |
| :------------------ | :-------------- | :------------------- |
| **VIN**             | **3.3V**        | 3.3V Power Supply    |
| **GND**             | **GND**         | Ground               |
| **SDA**             | **GPIO 21**     | I2C Data Line        |
| **SCL**             | **GPIO 22**     | I2C Clock Line       |
| **INT**             | _Not Connected_ | Interrupt (optional) |

### Serial Configuration:

- **Baud Rate**: `921600` (or `115200`)
- **Format**: `sample_index,expected_timestamp_ms,host_timestamp_ms,ir,red`
- **Sampling Rate**: $100\text{ Hz}$

---

## Installation & Quickstart Guide

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Linux / macOS / Windows with CUDA support for accelerated inference

### Step 1: Clone Repository

```bash
git clone https://github.com/omkar703/Cuffless-BP-Model.git
cd Cuffless-BP-Model
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional for LLM Explainer)

Copy the example environment file and add your Groq API key (if you wish to enable the online LLaMA 3.3 70B explainer; a deterministic local fallback is built-in):

```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### Step 5: Launch the Streamlit Application

```bash
streamlit run code/phase6c_app/app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## Running Unit Tests & Verifications

The scientific integrity, frozen model invariance, causal DSP pipeline, and reliability thresholds are covered by unit test suites:

```bash
# Test 1: Live Experiment streaming pipeline (6 tests)
PYTHONPATH=code:code/phase6c_app:code/scripts:code/phase4a python code/tests/test_live_experiment.py

# Test 2: Phase 7 Deterministic Reliability Engine & Frozen Weights (18 tests)
PYTHONPATH=code python code/phase7_reliability/test_phase7_reliability.py
```

Expected output: **All 24 unit tests pass with zero errors**.

---

## Repository Structure

```
Cuffless-BP-Model/
├── .streamlit/                      # Streamlit configuration and themes
│   └── config.toml
├── code/
│   ├── notebooks/                   # Documented research notebooks (Phase 5A, 5B, 5C, 6A, 6B)
│   ├── outputs/
│   │   ├── phase4a_single_model/    # Frozen Phase 4A CNN checkpoint
│   │   ├── phase4b_temporal_gru/    # Frozen Phase 4B GRU checkpoint
│   │   ├── phase5c_extreme_aware/   # Isotonic calibration models & conformal quantiles
│   │   ├── phase7_reliability/      # Fitted reliability engine pickle model
│   │   └── windows/                 # Quality-controlled window manifests
│   ├── phase4a/                     # Phase 4A model definitions & dataset loaders
│   ├── phase6c_app/                 # Streamlit presentation & experiment interface
│   │   ├── analysis/                # Live streaming engine & hardware audit
│   │   ├── data_io/                 # Recorded session loaders & export generators
│   │   ├── ui/                      # Guided steps, Live Experiment UI, components & theme
│   │   ├── app.py                   # Streamlit entrypoint
│   │   └── config.py
│   ├── phase7_reliability/          # Reliability engine logic & unit test suite
│   ├── scripts/                     # Causal DSP, resamplers, and reliability scripts
│   └── tests/                       # Live experiment end-to-end unit tests
├── hardware/
│   ├── firmware/                    # ESP32 Arduino C++ firmware for MAX30102
│   └── samples/                     # Verified multi-session physical recordings (.zip)
├── dataset_info.md                  # Kaggle dataset reference & citation metadata
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## Dataset & References

1. **Benchmark Dataset**: [Kaggle / UCI Machine Learning Repository Blood Pressure Dataset](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset)
2. **Foundational Citations**:
   - M. Kachuee, M. M. Kiani, H. Mohammadzadeh, M. Shabany, _"Cuff-Less Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring"_, IEEE Transactions on Biomedical Engineering (TBME), 2016.
   - M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, _"Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time"_, IEEE International Symposium on Circuits and Systems (ISCAS), 2015.
   - Association for the Advancement of Medical Instrumentation (AAMI) / ANSI / ISO 81060-2:2019 non-invasive sphygmomanometers protocol.
   - British Hypertension Society (BHS) Standard for Cuffless Blood Pressure Evaluation.

---
