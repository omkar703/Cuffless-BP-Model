# Existing Blood Pressure Dataset Analysis

## 1. Project Overview

This document presents an exhaustive, research-oriented analysis of the existing blood-pressure estimation workspace located at `/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone`.

The ultimate objective of the project is to build a scientifically rigorous, cuffless blood-pressure estimation pipeline that estimates **Systolic Blood Pressure (SBP)** and **Diastolic Blood Pressure (DBP)** using **Photoplethysmography (PPG)** alone, eventually interfacing with real wearable sensor hardware (**ESP32 + MAX30102**).

In accordance with strict research principles:
- **No new models were trained during this phase.**
- **No original files were deleted, overwritten, or modified.**
- **No external packages were installed.**
- **Existing notebooks were critically audited rather than assumed to be correct.**

The goal of this analysis is to deconstruct every existing dataset file, notebook, signal flow, model architecture, evaluation metric, and methodological vulnerability (notably data leakage, calibration reliance, and ECG dependencies) to lay a rock-solid foundation for subsequent pipeline design.

---

## 2. Dataset Structure

### 2.1 Origin and Context
The dataset present in the workspace is the widely studied benchmark:
- **Kaggle**: *Blood Pressure Dataset / Cuff-Less Blood Pressure Estimation*
- **Curator**: Mohamad Kachuee et al.
- **Underlying Source**: PhysioNet MIMIC-II Waveform Database.
- **Reference Publications**:
  1. M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, *"Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time"*, IEEE ISCAS, 2015.
  2. M. Kachuee, M. M. Kiani, H. Mohammadzadeh, M. Shabany, *"Cuff-Less Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring"*, IEEE TBME, 2016.

### 2.2 Physical File Layout in Workspace
```
/run/media/op/DATA/Omkar/VIT/4y/sem2/Capstone/
├── BloodPressureDataset/
│   ├── part_1.mat               (450 MB)
│   ├── part_2.mat               (391 MB)
│   ├── part_3.mat               (304 MB)
│   ├── part_4.mat               (427 MB)
│   ├── part_5.mat               (462 MB)
│   ├── part_6.mat               (439 MB)
│   ├── part_7.mat               (208 MB)
│   ├── part_8.mat               (327 MB)
│   ├── part_9.mat               (435 MB)
│   ├── part_10.mat              (411 MB)
│   ├── part_11.mat              (418 MB)
│   ├── part_12.mat              (341 MB)
│   └── Samples/
│       ├── rec_1.csv            (1.3 MB)
│       ├── rec_2.csv            (1.3 MB)
│       └── ... rec_500.csv      (total 500 CSV files)
├── sample notebooks/
│   ├── abp-estimation.ipynb
│   ├── blood-pressure-2-trial-2.ipynb
│   ├── bloodpressure-abp.ipynb
│   ├── bloodpressure-analysis.ipynb
│   ├── cnn-lstm.ipynb
│   ├── comparing-models-to-predict-bp-from-ppg.ipynb
│   ├── ecg-ppg-ppg.ipynb
│   ├── health-montoring.ipynb
│   ├── ml-bp.ipynb
│   └── predict-min-max-blood-pressurre-using-ecg.ipynb
├── code/                         (currently empty directory)
├── dataset_info.md               (dataset metadata and citations)
└── Readme.md                     (Kaggle source link)
```

### 2.3 File Format & Internal Representation
1. **MATLAB Files (`part_1.mat` to `part_12.mat`)**:
   - Total binary size: ~4.58 GB.
   - Format: MATLAB v7 / v7.3 MAT-files readable via `scipy.io.loadmat` using top-level dictionary key `'p'`.
   - Data structure: A $1 \times N_{records}$ cell array (where $N_{records} \approx 1,000$ per file, totaling ~12,000 records across all 12 parts).
   - Each cell `p[0, i]` contains a 2D float matrix of shape `(3, N_samples)`:
     - **Row 0**: PPG signal (photoplethysmogram from fingertip probe, normalized optical sensor voltage), $F_s = 125\text{ Hz}$.
     - **Row 1**: ABP signal (invasive radial arterial catheter blood pressure in mmHg), $F_s = 125\text{ Hz}$.
     - **Row 2**: ECG signal (electrocardiogram from Lead II), $F_s = 125\text{ Hz}$.
   - The number of samples per record $N_{samples}$ varies by recording duration, commonly 61,000 samples (~8.13 minutes).

2. **CSV Files (`Samples/rec_1.csv` to `rec_500.csv`)**:
   - 500 individual CSV files extracted from initial parts of the dataset.
   - Format: Exactly 3 non-empty lines (rows) of comma-delimited numeric floats:
     - **Line 1**: PPG time-series values ($F_s = 125\text{ Hz}$).
     - **Line 2**: ABP time-series values ($F_s = 125\text{ Hz}$, in mmHg).
     - **Line 3**: ECG time-series values ($F_s = 125\text{ Hz}$).
   - Typical length: Exactly 61,000 values per channel (corresponding to $61,000 / 125 = 488\text{ seconds} \approx 8.13\text{ minutes}$).

---

## 3. Dataset Statistics

| Metric | Determined Value | Method of Determination / Source |
| :--- | :--- | :--- |
| **Number of MAT Files** | 12 files (`part_1.mat` to `part_12.mat`) | Directory listing |
| **Number of CSV Files** | 500 files (`rec_1.csv` to `rec_500.csv`) | Directory listing in `Samples/` |
| **Total Raw Records** | 12,000 records (~1,000 per MAT file) | Verified via `cnn-lstm.ipynb` and `ml-bp.ipynb` batch loaders |
| **Clean Kept Records** | 10,859 to 10,874 records | Records surviving quality filtering (NaN, flatline, extreme BP) |
| **Total Segments (2s window, 0 overlap)** | 1,333,867 segments | Computed across 12 parts in `cnn-lstm.ipynb` |
| **Total Segments (10s window, 50% overlap)** | 513,472 clean segments | Computed across 12 parts in `ml-bp.ipynb` |
| **Number of Unique Subjects** | **Could not determine from the available files.** | Records are anonymous. `dataset_info.md` notes multiple records per patient exist contiguously, but explicit subject IDs are absent. |
| **Sampling Frequency ($F_s$)** | Exactly **125 Hz** across all 3 channels | Confirmed via `dataset_info.md`, Kachuee papers, and signal FFT spectra |
| **Signal Duration per Record** | Up to 61,000 samples (488 s / ~8.13 min) | Direct inspection of `rec_*.csv` and MAT cells |
| **PPG Signal Range** | Uncalibrated optical voltage; typical values 0.5 to 3.2 V | Peak-to-peak amplitude ~0.1 to 2.0 units |
| **ABP / SBP Distribution** | Mean: **129.5 - 131.6 mmHg**, Std: **22.0 - 22.6 mmHg**, Range: **[62.0, 200.0] mmHg** | Empirically verified across 1.3M segments |
| **ABP / DBP Distribution** | Mean: **65.4 - 66.6 mmHg**, Std: **10.8 - 11.2 mmHg**, Range: **[40.0, 120.0] mmHg** | Empirically verified across 1.3M segments |
| **ABP / MAP Distribution** | Mean: **88.8 mmHg**, Std: **13.9 mmHg** | Computed from continuous ABP waveform integration |
| **Signal Quality Anomalies** | Flatlines (`std < 1e-6`), sensor saturation, catheter flush spikes (>250 mmHg), zero-line drops (<30 mmHg), NaN/Inf floats | Quantified in `cnn-lstm.ipynb` (approx. 5-8% segments rejected) |

---

## 4. Notebook Inventory

| # | Notebook Filename | Primary Approach / Paradigm | Model(s) | Signals Loaded | Splitting Strategy |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | `abp-estimation.ipynb` | Deep Learning on raw segments (2.0s) | Conv1D + LSTM + Dense | PPG + ECG $\to$ MAP | Random `train_test_split` |
| **2** | `blood-pressure-2-trial-2.ipynb` | Handcrafted features on 60s windows | LLS & Linear Regression | PPG $\to$ MAP / avg BP | Random `train_test_split` |
| **3** | `bloodpressure-abp.ipynb` | Point-by-point instantaneous scalar mapping | Linear Regression, MLP | PPG $\to$ ABP scalar | Random `train_test_split` |
| **4** | `bloodpressure-analysis.ipynb` | PTT & morphological features (1.0s) | Random Forest Regressor | PPG + ECG $\to$ SBP, DBP, MAP | Random `train_test_split` + 5-fold K-Fold |
| **5** | `cnn-lstm.ipynb` | End-to-end multi-output DL on 2.0s segments | Conv1D + LSTM + Dense | PPG + ECG $\to$ SBP + DBP | **GroupShuffleSplit** (by record) |
| **6** | `comparing-models-to-predict-bp-from-ppg.ipynb` | TPU-targeted comparative benchmark | LR, MLP, LSTM | PPG $\to$ ABP waveform | Random `train_test_split` (Cleared outputs) |
| **7** | `ecg-ppg-ppg.ipynb` | Handcrafted PTT & volume features | Random Forest Regressor | PPG + ECG $\to$ SBP, DBP | Random `train_test_split` + 5-fold K-Fold (Buggy) |
| **8** | `health-montoring.ipynb` | Synthetic / Toy simulation | SVR (on synthetic data) | Purely synthetic random data | Random `train_test_split` (Irrelevant) |
| **9** | `ml-bp.ipynb` | Feature engineering + Calibrated Delta ML | XGBoost, LightGBM, RF, Stacking | PPG + ECG $\to$ $\Delta$SBP, $\Delta$DBP | **GroupKFold** (10-fold by record) |
| **10** | `predict-min-max-blood-pressurre-using-ecg.ipynb` | End-to-end DL on 0.51s ECG segments | ResNet + BiLSTM | **ECG only** $\to$ SBP, DBP | `validation_split=0.01` (Random) |

---

## 5. Notebook-by-Notebook Analysis

### 5.1 `abp-estimation.ipynb`
- **Purpose**: Estimate Mean Arterial Pressure (MAP) from combined PPG and ECG windows using a deep neural network.
- **Dataset / Files Used**: `part_1.mat` (first 1,000 records).
- **Signals Used**:
  - PPG: Used as input channel 1.
  - ECG: Used as input channel 2.
  - ABP: Used strictly to derive target (MAP).
- **Input Representation**: Raw time-domain segments of shape `(N, 250, 2)` representing 2.0 seconds ($250 / 125\text{ Hz}$) of stacked PPG and ECG.
- **Target Representation**: Scalar MAP calculated via mean of ABP window: $y = \frac{1}{W}\sum_{t=1}^{W} \text{ABP}_t$. Shape `(N, 1)`.
- **Sampling Frequency**: 125 Hz.
- **Preprocessing**:
  - No bandpass filtering, no baseline wander removal.
  - Global `StandardScaler().fit(X_flat)` and `StandardScaler().fit(y)` applied **before** train/test split.
- **Windowing**: Non-overlapping rectangular window: `sample_size = 250` (2.0 seconds), step = 250.
- **Feature Extraction**: None (end-to-end representation learning).
- **Model Architecture**:
  - `Conv1D(filters=256, kernel=5, padding='same', relu)` $\to$ `BatchNormalization` $\to$ `MaxPool1D(2)` $\to$ `Dropout(0.2)`
  - `Conv1D(filters=128, kernel=3, padding='same', relu)` $\to$ `BatchNormalization` $\to$ `MaxPool1D(2)` $\to$ `Dropout(0.2)`
  - `LSTM(256, return_sequences=False)` $\to$ `BatchNormalization` $\to$ `Dropout(0.3)`
  - `Dense(256, relu)` $\to$ `BatchNormalization` $\to$ `Dropout(0.3)`
  - `Dense(128, relu)` $\to$ `BatchNormalization`
  - `Dense(1, linear)`
- **Training Configuration**: Loss: MSE; Optimizer: Adam (`lr=5e-4`); Batch size: 64; Epochs: 25.
- **Evaluation & Results**:
  - Target: MAP only.
  - Test RMSE: **5.03 mmHg**
  - Test $R^2$: **0.8616**
  - SBP MAE / DBP MAE: **N/A** (not predicted).
- **Critical Flaws**:
  1. *ECG Dependency*: Cannot run on PPG alone.
  2. *Record Leakage*: 250-sample segments from the same 8-minute recording are randomly shuffled between train and test.
  3. *Scaler Leakage*: `StandardScaler` fitted on entire dataset prior to splitting.
  4. *Clinical Incompleteness*: Predicts only MAP, omitting critical diagnostic markers SBP and DBP.

---

### 5.2 `blood-pressure-2-trial-2.ipynb`
- **Purpose**: Test linear least squares and linear regression to predict mean blood pressure from PPG heart rate and pulse amplitude.
- **Dataset / Files Used**: `part_1.mat` (first 1,000 records).
- **Signals Used**: PPG (analyzed), ABP (target), ECG (extracted in loop, then discarded).
- **Input Representation**: 2 handcrafted scalar features per 60-second window:
  1. $\ln(\text{Heart Rate})$ from peak counts.
  2. $\ln(\text{mean Normal Pulse Volume})$ from peak-to-trough amplitude.
- **Target Representation**: $\ln(\text{average ABP})$ or $\ln(\text{MAP})$ over 60 seconds (7,500 samples).
- **Sampling Frequency**: 125 Hz.
- **Preprocessing**: None beyond Billauer's `peakdet` algorithm ($\delta = 0.3$).
- **Windowing**: Non-overlapping 60-second windows ($W = 7,500$ samples), total 500 segments.
- **Model Architecture**:
  1. Direct Analytical Linear Least Squares: $\mathbf{w} = (\mathbf{B}^T\mathbf{B})^{-1}\mathbf{B}^T\mathbf{A}$.
  2. Scikit-learn `LinearRegression()`.
- **Training Configuration**: Fit on all 500 samples or 70/30 split.
- **Evaluation & Results**:
  - Test $R^2$: **0.0860** (practically zero explanatory power).
  - SBP / DBP MAE: **N/A**.
- **Critical Flaws**:
  1. *Code Bugs*: In Cell 20, loop `for t in range(20)` references outer variable `series` instead of iterating window `t`. In Cell 21, error calculation divides by loop index variable `i`.
  2. *Model Collapse*: $R^2 < 0.09$ demonstrates that two static scalar features ($\ln(\text{HR})$ and amplitude) are utterly incapable of linear BP tracking across heterogeneous records.

---

### 5.3 `bloodpressure-abp.ipynb`
- **Purpose**: Attempt to map instantaneous PPG voltage directly to instantaneous ABP voltage.
- **Dataset / Files Used**: `part_1.mat` (first 1,000 records).
- **Signals Used**: PPG (input), ABP (target), ECG (plotted only).
- **Input Representation**: Single scalar float per time step: $X \in \mathbb{R}^{32,061,000 \times 1}$.
- **Target Representation**: Instantaneous arterial pressure at the exact same time step: $y \in \mathbb{R}^{32,061,000 \times 1}$.
- **Sampling Frequency**: 125 Hz.
- **Preprocessing**: None.
- **Windowing**: Window size = 1 sample ($1/125\text{ s} = 8\text{ ms}$). No temporal context whatsoever.
- **Model Architecture**:
  - Model 1: `LinearRegression()` (scalar input $\to$ scalar output).
  - Model 2: Dense MLP: `Dense(1024, relu)` $\to$ `Dropout(0.5)` $\to$ `Dense(512, relu)` $\to$ `Dropout(0.5)` $\to$ `Dense(64, relu)` $\to$ `Dense(1)`.
- **Training Configuration**: Train on first 1,000,000 random samples; Batch size: 128; Epochs: 5.
- **Evaluation & Results**:
  - Linear Regression 5-fold CV RMSE: **27.34 mmHg**
  - Linear Regression Test RMSE: **27.36 mmHg**
  - Neural Net Test RMSE: **25.90 mmHg**
  - Prediction sanity test (`model.predict([30])`): Outputs **343.4 mmHg** (absurd extrapolation).
- **Critical Flaws**:
  1. *Fatal Formulation Error*: A single instantaneous photoplethysmogram voltage contains zero phase, velocity, or morphological information. Mapping $V_{\text{PPG}}(t) \to P_{\text{ABP}}(t)$ pointwise is biologically and mathematically invalid.
  2. *Catastrophic Leakage*: 32 million points randomly partitioned by `train_test_split`.

---

### 5.4 `bloodpressure-analysis.ipynb`
- **Purpose**: Extract physiological Pulse Transit Time (PTT) and pulse contour features to estimate SBP, DBP, and MAP via Random Forest.
- **Dataset / Files Used**: `part_1.mat` (first 1,000 records).
- **Signals Used**:
  - ECG: Essential (used to detect R-peaks for PTT and RR intervals).
  - PPG: Essential (used to detect systolic peaks, dicrotic notches, foot points).
  - ABP: Target generation.
- **Input Representation**: 7 handcrafted physiological features per segment:
  1. $\text{PTT}_p$: R-peak to PPG maximum (peak).
  2. $\text{PTT}_f$: R-peak to PPG minimum (foot).
  3. $\text{PTT}_d$: R-peak to maximum PPG derivative ($VPG_{\max}$).
  4. $\text{PTT}_{\text{mean}}$: Average of $\text{PTT}_p, \text{PTT}_f, \text{PTT}_d$.
  5. $\text{AI}$: Augmentation Index ($P_2 / P_1$).
  6. $\text{LASI}$: Large Artery Stiffness Index (inter-peak time).
  7. $\text{HR}$: Heart rate from ECG RR interval ($60 / \Delta t_{RR}$).
- **Target Representation**: Multi-output target vector $\mathbf{y} = [\text{MAP}, \text{SBP}, \text{DBP}]$:
  - $\text{MAP} = \text{mean}(\text{ABP}_{\text{win}})$
  - $\text{SBP} = \max(\text{ABP}_{\text{win}})$
  - $\text{DBP} = \min(\text{ABP}_{\text{win}})$
- **Sampling Frequency**: 125 Hz.
- **Preprocessing**:
  - ECG: 4th-order Butterworth bandpass $[0.5, 40]\text{ Hz}$ via `filtfilt`.
  - PPG: 4th-order Butterworth lowpass with cutoff $8.0\text{ Hz}$ via `filtfilt`.
- **Windowing**: Non-overlapping windows of `sample_size = 125` (1.0 second).
- **Model Architecture**: Scikit-learn `RandomForestRegressor()`.
- **Training Configuration**: 5-fold cross-validation (`KFold(n_splits=5, shuffle=True)`).
- **Evaluation & Results**:
  - 5-fold CV Average RMSE: **13.46 mmHg** (across all 3 targets).
  - Test Set RMSE: **13.28 mmHg**, STD of error: **13.28 mmHg**.
- **Critical Flaws**:
  1. *ECG Dependency*: 5 of 7 features ($\text{PTT}_p, \text{PTT}_f, \text{PTT}_d, \text{PTT}_{\text{mean}}, \text{HR}$) require ECG R-peaks. **Cannot be deployed as a PPG-only system.**
  2. *Window Length Problem*: At $W = 1.0\text{ s}$ (125 samples), for any heart rate below 60 bpm, a 1-second window contains at most one peak, causing feature extraction to return `None` and drop segments.
  3. *Leakage*: Segments partitioned via random K-Fold without grouping by record.

---

### 5.5 `cnn-lstm.ipynb`
- **Purpose**: Comprehensive end-to-end deep learning pipeline predicting continuous SBP and DBP directly from multi-channel raw waveforms across all 12 dataset parts with patient-level splitting.
- **Dataset / Files Used**: **ALL 12 MAT files** (`part_1.mat` through `part_12.mat`).
- **Signals Used**:
  - PPG: Channel 0 of input tensor.
  - ECG: Channel 1 of input tensor.
  - ABP: Target generation.
- **Input Representation**: Shape `(N, 250, 2)` (250 time samples, 2 channels).
- **Target Representation**: Continuous joint vector $[\text{SBP}, \text{DBP}] \in \mathbb{R}^{N \times 2}$ where:
  - $\text{SBP} = \max(\text{ABP}_{\text{segment}})$
  - $\text{DBP} = \min(\text{ABP}_{\text{segment}})$
- **Sampling Frequency**: 125 Hz.
- **Preprocessing & Cleaning Pipeline**:
  - Comprehensive quality control:
    1. Rejection of NaN / Inf in any channel.
    2. Rejection of flatline signals ($\sigma < 10^{-6}$).
    3. Physiological BP bounds checking: $50 \le \text{SBP} \le 250\text{ mmHg}$, $30 \le \text{DBP} \le 150\text{ mmHg}$, and $\text{SBP} > \text{DBP} + 10$.
    4. PPG peak-to-peak amplitude validation: $10^{-4} \le \text{ptp} \le 10^6$.
  - Normalization: **Per-segment, per-channel Z-score**:
    $$\tilde{X}_{n, t, c} = \frac{X_{n, t, c} - \mu_{n, c}}{\sigma_{n, c} + 10^{-8}}$$
    *(Strictly causal and independent per window; zero leakage across samples).*
- **Windowing**: Non-overlapping rectangular window: `sample_size = 250` (2.0 seconds).
- **Train/Test Splitting**: **`GroupShuffleSplit` by `record_id`** (70% train, 30% test).
  - Train: 8,400 records (935,124 segments).
  - Test: 3,600 records (398,743 segments).
  - Record overlap: **Exactly 0**.
- **Model Architecture**:
  - `Conv1D(256, kernel=5, padding='same', relu)` $\to$ `BN` $\to$ `MaxPool(2)` $\to$ `Dropout(0.2)`
  - `Conv1D(128, kernel=3, padding='same', relu)` $\to$ `BN` $\to$ `MaxPool(2)` $\to$ `Dropout(0.2)`
  - `LSTM(256, return_sequences=False)` $\to$ `BN` $\to$ `Dropout(0.3)`
  - `Dense(256, relu)` $\to$ `BN` $\to$ `Dropout(0.3)`
  - `Dense(128, relu)` $\to$ `BN`
  - `Dense(2, linear)`
- **Training Configuration**: Loss: MSE; Optimizer: Adam (`lr=5e-4`, `clipnorm=1.0`); Batch size: 64; Epochs: 25; Full checkpointing and resumption mechanism.
- **Evaluation & Results**:
  - Tested on 398,743 unseen segments from 3,600 held-out records:
  - **SBP Results**:
    - **MAE**: **9.20 mmHg**
    - **RMSE**: **13.50 mmHg**
    - **$R^2$**: **0.6449**
    - **Mean Error (Bias)**: **-1.04 mmHg**
    - **STD of Error**: **13.46 mmHg**
    - *AAMI Standard check*: Bias $\le 5$ (Passed), STD $\le 8$ (Failed).
  - **DBP Results**:
    - **MAE**: **4.84 mmHg**
    - **RMSE**: **7.64 mmHg**
    - **$R^2$**: **0.5230**
    - **Mean Error (Bias)**: **-0.64 mmHg**
    - **STD of Error**: **7.61 mmHg**
    - *AAMI Standard check*: Bias $\le 5$ (**Passed**), STD $\le 8$ (**Passed**).
- **Critical Flaws**:
  1. *Dual-signal dependence*: Model expects `(250, 2)` (PPG + ECG). Cannot infer from PPG alone without retraining/architecture change.
  2. *Target Scaling*: `StandardScaler` for $y$ fitted globally before `GroupShuffleSplit` (minor leakage).
  3. *Window Extrema Target*: In 2.0 seconds, taking $\max$ and $\min$ of ABP can mischaracterize BP if an incomplete pulse or baseline shift occurs.

---

### 5.6 `comparing-models-to-predict-bp-from-ppg.ipynb`
- **Purpose**: Benchmark Linear Regression, Multi-Layer Perceptron (ANN), and LSTM models on Google Cloud TPU to reconstruct ABP waveforms from PPG.
- **Dataset / Files Used**: `part_1.mat`.
- **Signals Used**: PPG and ABP.
- **Input Representation**: PPG time slices.
- **Target Representation**: Continuous ABP waveform points.
- **Status of Results**: **Outputs Cleared / Not Executed in Saved File.**
  - All code blocks exist, including TPU cluster initialization and layer definitions, but cell execution outputs were stripped prior to committing.

---

### 5.7 `ecg-ppg-ppg.ipynb`
- **Purpose**: Extract ECG and PPG time-domain and frequency-domain features to predict SBP and DBP using Random Forest.
- **Dataset / Files Used**: `part_1.mat`.
- **Signals Used**: PPG, ECG, ABP.
- **Features Extracted**: PTT, HeartRate, SysVolChange, SysMaxVolChange, DiaMaxVolChange, PWIR, HRV, SDNN, RMSSD (9 features).
- **Reported Results**:
  - 5-fold CV SBP MAE: **3.53 ± 0.11 mmHg**, STD: 3.40 mmHg.
  - 5-fold CV DBP MAE: **3.29 ± 0.12 mmHg**, STD: 3.41 mmHg.
- **CRITICAL IMPLEMENTATION BUG & RESULT INVALIDATION**:
  - In Cell 10, feature extraction computes RMSSD and HRV requiring at least 4 ECG peaks in a 1-second window ($W = 125$). Because a normal human heart rate cannot fit 4 beats into 1.0 second, these features produce `NaN` for almost all rows.
  - The notebook executes:
    ```python
    X = pd.DataFrame(features_dict).dropna()
    y_sbp = sbp[: len(X)]
    y_dbp = dbp[: len(X)]
    ```
  - **Fatal Bug**: Calling `.dropna()` on `X` removed thousands of sparse rows, but `y_sbp` and `y_dbp` were **not filtered by index**! Instead, it truncated the first `len(X)` entries from the unaligned target array!
  - **Conclusion**: Features from row $k$ were paired with target values from row $k$ of the original dataset, completely desynchronizing signals from their ground truth. **The reported 3.53 mmHg MAE is a mathematical artifact of complete index misalignment and must be entirely rejected.**

---

### 5.8 `health-montoring.ipynb`
- **Purpose**: Purported multi-disease and vitals monitoring demo (Diabetes, Arrhythmia, Stress, Blood Pressure).
- **Dataset / Files Used**: None of the local project data files.
- **Code Reality**:
  - Implements `simulate_dataset()` using `np.random.normal()` and synthetic formulas:
    $$\text{sbp} = 110 + 0.5\cdot\text{age} + 0.3\cdot\text{bmi} - 0.2\cdot\text{ptt}\cdot 1000$$
  - Trains an `SVR()` on fake numbers and reports `BP Model MAE: 7.89 mmHg`.
- **Conclusion**: A toy script containing zero real physiological signals. **Must NOT be reused.**

---

### 5.9 `ml-bp.ipynb`
- **Purpose**: State-of-the-art machine learning benchmark on 10-second segments incorporating extensive pulse morphology, derivative curves (VPG, APG), and calibrated delta modeling with 10-fold GroupKFold.
- **Dataset / Files Used**: **ALL 12 MAT files** (`part_1.mat` to `part_12.mat`).
- **Signals Used**: PPG, ECG, ABP. Total 513,472 clean segments across 10,859 records.
- **Input Representation**: 22 engineered features per 10-second segment (1250 samples, step 625 = 50% overlap):
  - *PPG-Only Morphology & Derivatives (13 features)*:
    1. `vpg_slope`: Maximum slope of Velocity Plethysmogram (1st derivative).
    2. `crest_time`: Time from pulse foot to systolic peak.
    3. `sys_area`: Systolic area under PPG curve.
    4. `crest_ratio`: Ratio of crest time to total pulse period.
    5. `si_proxy`: Stiffness Index proxy ($T / (T - T_c)$).
    6. `apg_ba`: Acceleration Plethysmogram $b/a$ ratio (2nd derivative dicrotic notch indicator).
    7. `diastolic_time`: Diastolic decay duration ($T - T_c$).
    8. `pulse_period`: Duration of cardiac pulse cycle.
    9. `width_50`: Pulse width at 50% amplitude height.
    10. `spec_cardiac`: Normalized spectral power in cardiac band (0.5 - 3.0 Hz).
    11. `ppg_skew`: Statistical skewness of pulse contour.
    12. `hr_mean`: Heart rate derived from PPG inter-beat intervals.
    13. `hr_x_crest`: Interaction term ($\text{HR} \times \text{crest\_time}$).
  - *ECG-Dependent Features (9 features)*:
    14. `pat_foot`: Pulse Arrival Time to pulse onset ($R_{\text{ECG}} \to \text{foot}_{\text{PPG}}$).
    15. `pat_peak`: Pulse Arrival Time to peak ($R_{\text{ECG}} \to \text{peak}_{\text{PPG}}$).
    16. `pat_tangent`: Pulse Arrival Time to maximum tangent slope.
    17. `ecg_mobility`: Hjorth mobility of ECG.
    18. `ecg_complexity`: Hjorth complexity of ECG.
    19. `ecg_entropy`: Spectral entropy of ECG.
    20. `ecg_autocorr`: Autocorrelation of ECG.
    21. `hr_x_pat`: Interaction term ($\text{HR} \times \text{PAT}$).
    22. `ecg_mob_x_pat`: Interaction term ($\text{ECG mobility} \times \text{PAT}$).
- **Target Generation (`extract_bp_from_abp`)**:
  - Detects true arterial systolic peaks using `find_peaks(abp, distance=50, prominence=10)`.
  - Identifies true diastolic troughs between adjacent peaks using `argmin`.
  - Computes physiological mean SBP and mean DBP across all validated beats in the 10-second segment.
  - **This is by far the most medically sound ground-truth generation in the workspace.**
- **Modeling Paradigm & Experimental Setup**:
  - Evaluates: XGBoost, LightGBM, Random Forest, Ridge Regression, ElasticNet, and Weighted Stacking.
  - Splitting: Strict **`GroupKFold(n_splits=10)` by `record_id`**.
- **The "Calibrated Delta Model" Mechanism ($N_{\text{CAL}}$)**:
  - For each subject/record, the algorithm reserves the first $N_{\text{CAL}}$ segments ($N_{\text{CAL}} = 10$ or $15$, i.e., 50 to 75 seconds) as **calibration baseline**.
  - It extracts the patient's **actual true arterial blood pressure** during these calibration segments:
    $$\text{cal}_s = \text{median}(y_{\text{train\_cal}, \text{SBP}}), \quad \text{cal}_d = \text{median}(y_{\text{train\_cal}, \text{DBP}})$$
  - The ML model is trained **only to predict intra-subject variations (residuals)**:
    $$\Delta y = y - \text{cal\_bp}$$
  - The final prediction is:
    $$\hat{y} = \text{cal\_bp} + \hat{\Delta y}_{\text{model}}$$
- **Evaluation & Results**:
  - *Baseline Calibration Only (`cal_only` - no ML at all, just predicting the patient's past mean BP)*:
    - **SBP MAE**: **5.33 - 5.46 mmHg**, STD: 8.40 mmHg.
    - **DBP MAE**: **2.88 - 2.96 mmHg**, STD: 4.75 mmHg.
  - *Full Delta Model (XGBoost with $N_{\text{CAL}}=15$)*:
    - **SBP MAE**: **4.96 mmHg**, RMSE: 7.81 mmHg, $R^2$: 0.8726, Bias: +0.04 mmHg.
    - **DBP MAE**: **2.60 mmHg**, RMSE: 4.36 mmHg, $R^2$: 0.8295, Bias: +0.03 mmHg.
    - Meets AAMI criteria ($\le 5\text{ mmHg}$ MAE, $\le 8\text{ mmHg}$ STD).
  - *Incremental Contribution of ML over Patient Baseline*:
    - For SBP: $\Delta\text{MAE} = 5.33 - 4.96 = \mathbf{0.37\text{ mmHg}}$ improvement.
    - For DBP: $\Delta\text{MAE} = 2.88 - 2.60 = \mathbf{0.28\text{ mmHg}}$ improvement.
- **Critical Methodological Takeaway**:
  - The low MAE ($<5\text{ mmHg}$) is **93-95% driven by knowing the subject's ground-truth arterial blood pressure during the calibration period**.
  - The machine learning model only accounts for a tiny $0.3\text{ mmHg}$ adjustment.
  - **This is a calibration-dependent system, not a calibration-free cuffless estimator.**

---

### 5.10 `predict-min-max-blood-pressurre-using-ecg.ipynb`
- **Purpose**: Predict SBP and DBP from ECG waveforms using a ResNet-BiLSTM architecture.
- **Dataset / Files Used**: `part_1.mat` to `part_6.mat`.
- **Signals Used**: **ECG and ABP only. PPG is completely omitted.**
- **Windowing**: $W = 64$ samples ($64 / 125\text{ Hz} = 0.512\text{ seconds}$).
- **Target Representation**: $\min(\text{ABP}_{64})$ as DBP, $\max(\text{ABP}_{64})$ as SBP.
- **Model Architecture**: ResNet blocks (Conv1D + ReLU + BN + Residual Add) + Bidirectional LSTM + Dense output branches with custom Huber-like `HyperLoss`.
- **Validation Results**: Validation SBP MAE: **7.25 mmHg**, DBP MAE: **4.15 mmHg**.
- **Critical Flaws**:
  1. *Zero PPG relevance*: Does not utilize PPG at all.
  2. *Unphysiological windowing*: In 0.51 seconds, a complete cardiac cycle rarely occurs, rendering the window maximum and minimum meaningless as true systolic and diastolic values.
  3. *Leakage*: `validation_split=0.01` performed randomly across time steps.

---

## 6. PPG Pipeline

### 6.1 Loading
- PPG is read from row index 0 of MATLAB cells: `rec[0]` or line 1 of `rec_*.csv`.
- Signal values represent raw optical photoplethysmographic reflection/transmission voltages recorded at fingertips via pulse oximeter probes in MIMIC-II.

### 6.2 Sampling Frequency
- **125 Hz** across all files and records.
- Nyquist limit: $62.5\text{ Hz}$.

### 6.3 Filtering Implementations Found
1. **No Filter / Raw**: Used in `abp-estimation.ipynb`, `cnn-lstm.ipynb`, `bloodpressure-abp.ipynb`.
   - *Risk*: Suffers from baseline wander (respiration, venous pooling) and high-frequency quantization noise.
2. **Butterworth Lowpass (4th order, $8.0\text{ Hz}$)**: Used in `bloodpressure-analysis.ipynb`.
   - Attenuates high-frequency noise while preserving systolic peak and dicrotic notch.
3. **Butterworth Bandpass ($[0.5, 8.0]\text{ Hz}$, 3rd order)**: Implemented via bidirectional zero-phase `filtfilt` in `ml-bp.ipynb`.
   - Cuts low-frequency baseline drift below 0.5 Hz (30 bpm) and removes electrical/ambient interference above 8 Hz (harmonics above cardiac fundamental).
   - *Assessment*: This is the most optimal filter found in the workspace.

### 6.4 Normalization Methods Found
1. **Global Dataset-wide StandardScaler**: `abp-estimation.ipynb`.
   - Fits mean and variance across train + test sets simultaneously $\to$ **Direct Data Leakage**.
2. **Pointwise Min-Max**: `comparing-models-to-predict-bp-from-ppg.ipynb`.
   - Flattens physiological pulse amplitude differences across patients.
3. **Per-Segment, Per-Channel Z-Score**: `cnn-lstm.ipynb`.
   - For every isolated segment: $\tilde{x}(t) = \frac{x(t) - \mu_{\text{seg}}}{\sigma_{\text{seg}}}$.
   - Preserves intra-beat relative morphological dynamics while stabilizing numerical gradients without inter-sample leakage.
4. **RobustScaler inside CV**: `ml-bp.ipynb`.
   - Uses median and interquartile range (IQR) fitted strictly on training folds.

### 6.5 Pulse / Beat Detection
- **`scipy.signal.find_peaks`**:
  - `distance = int(0.4 * fs)` ($\approx 50$ samples = 400 ms, corresponding to max detectable HR of 150 bpm).
  - Prominence filtering ($0.02$ to $10.0$) ensures noise ripples are not flagged as systolic peaks.
- **Billauer's `peakdet`**: Used in `blood-pressure-2-trial-2.ipynb` ($\delta = 0.3$).

### 6.6 PPG Derivative Curves (VPG & APG)
- Found exclusively in `ml-bp.ipynb`:
  - **Velocity Plethysmogram (VPG)**: First numerical derivative $v(t) = \frac{d}{dt} x(t)$. Captures maximum systolic ejection velocity (`vpg_slope`).
  - **Acceleration Plethysmogram (APG)**: Second numerical derivative $a(t) = \frac{d^2}{dt^2} x(t)$. Captures characteristic $a, b, c, d, e$ waves used for vascular aging and arterial stiffness analysis ($b/a$ ratio).

---

## 7. ABP / Blood Pressure Ground Truth

### 7.1 How SBP, DBP, and MAP are Obtained Across Notebooks

```
                      ┌──────────────────────────────────────────────┐
                      │             Arterial Catheter ABP            │
                      └──────────────────────┬───────────────────────┘
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         │                                   │                                   │
         ▼                                   ▼                                   ▼
 [Single Point]                      [Window Extrema]                   [Beat-by-Beat Peaks]
 bloodpressure-abp                    cnn-lstm, anal.                        ml-bp.ipynb
                                    
 y = ABP(t)                          SBP = max(ABP_win)                 1. find_peaks(ABP, prom=10)
                                     DBP = min(ABP_win)                    -> sbp_beats
 (Physiologically invalid;          (Simple, but sensitive to           2. argmin between peaks
  no temporal context)               noise & baseline shifts)              -> dbp_beats
                                                                        3. SBP = mean(sbp_beats)
                                                                           DBP = mean(dbp_beats)
                                                                        (Scientifically robust)
```

1. **Instantaneous Pointwise ABP** (`bloodpressure-abp.ipynb`):
   - $y = \text{ABP}_t$. Invalid formulation.
2. **Window Mean (MAP)** (`abp-estimation.ipynb`, `blood-pressure-2-trial-2.ipynb`):
   - $\text{MAP} = \frac{1}{W}\sum_{t=1}^{W} \text{ABP}_t$. Accurate for MAP, but fails to provide SBP and DBP.
3. **Window Extrema ($\max / \min$)** (`cnn-lstm.ipynb`, `predict-min-max-blood-pressurre-using-ecg.ipynb`):
   - $\text{SBP} = \max_{t \in W} \text{ABP}_t$
   - $\text{DBP} = \min_{t \in W} \text{ABP}_t$
   - Simple and vectorized, but if a high-frequency motion spike or line disturbance enters the window, $\max$ picks up the artifact.
4. **Beat-by-Beat Physiological Detection** (`ml-bp.ipynb`):
   - Identifies local arterial pressure peaks with prominence thresholding $\ge 10\text{ mmHg}$.
   - Locates corresponding diastolic troughs.
   - Calculates segment average over individual cardiac cycles:
     $$\text{SBP}_{\text{seg}} = \frac{1}{K}\sum_{k=1}^{K} \text{Peak}_k, \quad \text{DBP}_{\text{seg}} = \frac{1}{K-1}\sum_{k=1}^{K-1} \text{Trough}_k$$
   - Eliminates anomalous beats where $\text{DBP} \ge \text{SBP}$.
   - **Recommended standard for all future ground truth generation.**

---

## 8. ECG Usage Audit

| Notebook | ECG Status | Description of ECG Role | Can Run PPG-Only As-Is? |
| :--- | :---: | :--- | :---: |
| `abp-estimation.ipynb` | **USED AS MODEL INPUT** | Stacked as Channel 2 of input tensor `(250, 2)` | **NO** |
| `blood-pressure-2-trial-2.ipynb` | **NOT USED** | Extracted in data load loop, but omitted from models | **YES** |
| `bloodpressure-abp.ipynb` | **NOT USED** | Plotted for visualization; excluded from model input | **YES** |
| `bloodpressure-analysis.ipynb` | **USED TO DERIVE FEATURES** | ECG R-peaks required to compute PTTp, PTTf, PTTd, HR | **NO** |
| `cnn-lstm.ipynb` | **USED AS MODEL INPUT** | Stacked as Channel 2 of input tensor `(250, 2)` | **NO** |
| `comparing-models-to-predict-bp-from-ppg.ipynb` | **NOT USED** | Input is PPG only | **YES** |
| `ecg-ppg-ppg.ipynb` | **USED TO DERIVE FEATURES** | ECG required for PTT, HRV, SDNN, RMSSD | **NO** |
| `health-montoring.ipynb` | **NOT USED** | Synthetic random data | **N/A** |
| `ml-bp.ipynb` | **USED TO DERIVE FEATURES** | 9 of 22 features (PAT, ECG Hjorth, entropy) require ECG | **NO** |
| `predict-min-max-blood-pressurre-using-ecg.ipynb` | **USED AS MODEL INPUT** | Trained exclusively on ECG; PPG not loaded | **NO (ECG-only)** |

### Implications for the Future PPG-Only Hardware System
- Every high-performing notebook in the existing workspace either feeds raw ECG directly into neural network channels or derives Pulse Transit Time (PTT/PAT) by measuring the delay between the ECG R-wave and the PPG systolic peak.
- In a wearable device with a MAX30102 sensor, **there are no ECG chest leads**.
- **Any feature or model architecture relying on ECG must be discarded or replaced with PPG-only morphology and pulse wave velocity proxies.**

---

## 9. Preprocessing Comparison

| Notebook | Signal Filtering | Normalization Method | Artifact / Quality Checks | Normalization Leakage? |
| :--- | :--- | :--- | :--- | :---: |
| `abp-estimation` | None (Raw) | `StandardScaler` on all $X$ and $y$ | None | **YES (Global Fit)** |
| `blood-pressure-2` | None | None | None | None |
| `bloodpressure-abp` | None | None | None | None |
| `bloodpressure-anal.` | ECG: Bandpass 0.5-40Hz; PPG: Lowpass 8Hz | None | Drops NaN feature rows | No |
| `cnn-lstm` | None (Raw) | **Per-segment Z-score** `(x-μ)/σ` | **Rigorous**: NaN/Inf, flatlines, BP bounds | **NO (Per-Segment)** |
| `comparing-models` | None | Min-Max $[0, 1]$ | None | Cleared |
| `ecg-ppg-ppg` | None | None | `dropna()` (Caused index bug) | No |
| `health-montoring` | None | `StandardScaler` | None | N/A |
| `ml-bp` | **PPG: Bandpass 0.5-8Hz; ECG: Bandpass 0.5-40Hz** | `RobustScaler` inside folds | **Rigorous**: NaN/Inf, flatlines, BP bounds | **NO (Inside CV)** |
| `predict-min-max` | None | Min-Max $[0, 1]$ | None | Unclear |

---

## 10. Windowing Comparison

| Notebook | Window Length ($W$) | Time Duration | Overlap / Step | Complete Cardiac Cycles Captured? | Suitability for BP Estimation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `predict-min-max` | 64 samples | 0.51 s | 0% (step 64) | **Rarely (<1 cycle)** | **Severely flawed** (cannot capture pulse extremes) |
| `bloodpressure-abp` | 1 sample | 0.008 s | 0% | **0 cycles** | **Fatal flaw** (no waveform context) |
| `bloodpressure-anal.` | 125 samples | 1.00 s | 0% (step 125) | Borderline (1 cycle at 60 bpm) | Poor (drops windows if HR < 60 bpm) |
| `ecg-ppg-ppg` | 125 samples | 1.00 s | 0% (step 125) | Borderline (1 cycle at 60 bpm) | Poor (cannot compute HRV/RMSSD in 1s) |
| `abp-estimation` | 250 samples | 2.00 s | 0% (step 250) | Yes (2 - 4 cycles) | Moderate (good for DL waveform models) |
| `cnn-lstm` | 250 samples | 2.00 s | 0% (step 250) | Yes (2 - 4 cycles) | Moderate (good for DL waveform models) |
| `ml-bp` | **1,250 samples** | **10.00 s** | **50% (step 625)** | **Yes (10 - 20 cycles)** | **Optimal** (robust beat averaging & morphology) |
| `blood-pressure-2` | 7,500 samples | 60.00 s | 0% (step 7500) | Yes (>60 cycles) | Too long (masks dynamic BP transients) |

---

## 11. Feature Extraction Comparison

| Approach | Notebooks | Extracted Features | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- |
| **None (End-to-End DL)** | `cnn-lstm`, `abp-estimation`, `predict-min-max` | Learned latent representations via Conv1D and LSTM | No hand-crafted feature bias; preserves subtle morphology | Requires larger datasets; black-box; computationally heavier for ESP32 edge deployment |
| **Pointwise Instantaneous** | `bloodpressure-abp` | Raw scalar voltage at time $t$ | Trivial | Completely invalid physiologically |
| **Minimal Scalar (HR + Amp)** | `blood-pressure-2` | $\ln(\text{HR}), \ln(\text{Amp})$ | Fast | Complete failure ($R^2 = 0.086$) |
| **PTT + Timing Morphology** | `bloodpressure-anal.` | PTTp, PTTf, PTTd, AI, LASI, HR | Grounded in pulse wave velocity theory | **Requires ECG** for PTT |
| **Comprehensive Multi-Domain** | `ml-bp` | 22 features: VPG slope, crest time, sys area, $b/a$ ratio, stiffness index proxy, pulse width 50%, skew, ECG Hjorth/entropy | High predictive power; clinically interpretable | 9 features require ECG leads |

---

## 12. Model Comparison

| Algorithm Category | Model Implementations Found | Notebooks | Strengths | Observed Failure Modes / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Ordinary Least Squares, Ridge, ElasticNet | `bloodpressure-abp`, `blood-pressure-2`, `ml-bp` | Fast, low complexity | Underfits non-linear vascular compliance; fails on raw points ($R^2 \approx 0.08$) |
| **Tree-based Ensembles** | `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor` | `bloodpressure-anal.`, `ecg-ppg-ppg`, `ml-bp` | Robust to outliers; handles non-linear feature interactions | Prone to memorizing patient baselines if split improperly |
| **Convolutional + Recurrent (CNN-LSTM)** | Conv1D + MaxPool + LSTM + Dense | `cnn-lstm`, `abp-estimation` | Automatically extracts temporal and morphological features | High parameter count; requires dual PPG+ECG input in existing code |
| **Residual Deep Networks** | ResNet1D + BiLSTM | `predict-min-max` | Strong feature abstraction | Overengineered for 0.5s windows; fails on ECG-only task |
| **Meta-Ensemble / Stacking** | Learned inverse-MAE weighted stacking | `ml-bp` | Combines tree bagging and boosting | Overkill when calibration baseline dominates variance |

---

## 13. Evaluation Comparison

| Evaluation Criterion | Scientific Best Practice (AAMI / BHS) | What Existing Notebooks Actually Did |
| :--- | :--- | :--- |
| **Data Splitting** | **Subject-level** (record-independent): No subject in train may appear in test. | **7 of 10 notebooks used random segment splits** (severe leakage). Only `cnn-lstm` and `ml-bp` grouped by record. |
| **Validation Standard** | **AAMI SP10**: Mean Error $\le 5\text{ mmHg}$, Standard Deviation $\le 8\text{ mmHg}$. | Only `cnn-lstm` and `ml-bp` formally computed bias and STD against AAMI. Others reported only RMSE/MAE. |
| **Calibration Status** | Truly **calibration-free** cuffless inference. | `ml-bp` claims SBP MAE 4.96 mmHg, but relies on **prior true arterial BP** for every subject. |
| **Target Outputs** | Both **Systolic (SBP)** and **Diastolic (DBP)** reported independently. | `abp-estimation` and `blood-pressure-2` predicted only MAP or average BP. |

---

## 14. Data Leakage & Methodological Vulnerabilities Audit

### 14.1 Record / Subject Overlap Leakage (Present in 7 of 10 Notebooks)
- **Where it occurs**: `abp-estimation.ipynb`, `blood-pressure-2-trial-2.ipynb`, `bloodpressure-abp.ipynb`, `bloodpressure-analysis.ipynb`, `comparing-models-to-predict-bp-from-ppg.ipynb`, `ecg-ppg-ppg.ipynb`, `predict-min-max-blood-pressurre-using-ecg.ipynb`.
- **Why it is leakage**: An 8-minute continuous recording from a single patient is chopped into hundreds of small windows (1s or 2s). Calling `train_test_split(shuffle=True)` places window $t$ in the training set and window $t+1$ (one second later) in the test set.
- **Artificial performance boost**: The model memorizes the patient's individual vascular stiffness, baseline blood pressure, and anatomical morphology. Test errors plummet artificially.
- **Correction**: Must partition exclusively via `GroupShuffleSplit` or `GroupKFold` using unique record/patient identifiers.

### 14.2 Preprocessing & Scaling Leakage
- **Where it occurs**: `abp-estimation.ipynb`.
- **Why it is leakage**: `StandardScaler.fit()` was executed on the full dataset array prior to splitting. Test set distribution parameters ($\mu, \sigma$) informed the training representation.
- **Correction**: Normalization statistics must be calculated strictly inside cross-validation training folds, or performed per-segment independently (`cnn-lstm.ipynb`).

### 14.3 The "Calibrated Delta" Trap
- **Where it occurs**: `ml-bp.ipynb`.
- **Why it is problematic**: To evaluate a test record, the model is provided the first 15 segments of ground truth invasive blood pressure ($\text{cal}_s, \text{cal}_d$).
- **Impact**: The patient's baseline BP alone yields SBP MAE $= 5.33\text{ mmHg}$ without running any ML model! The ML model only improves performance by $0.37\text{ mmHg}$.
- **Correction**: For a truly cuffless wearable system, models must be evaluated in a **strictly calibration-free protocol** where no initial arterial catheter readings are provided.

### 14.4 Alignment / Index Truncation Bug
- **Where it occurs**: `ecg-ppg-ppg.ipynb` (Cells 10-12).
- **Why it is fatal**: Dropping NaN rows from the feature dataframe without re-indexing the target array caused features from segment $k$ to be paired with blood pressure from segment $0, 1, 2...$. The reported MAE of 3.53 mmHg is invalid.

---

## 15. Existing Results Synthesis

| Notebook | Input Signals | Target | Model | SBP MAE (mmHg) | DBP MAE (mmHg) | RMSE (mmHg) | $R^2$ | Split Strategy | Methodological Validity |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `abp-estimation` | PPG + ECG | MAP | CNN-LSTM | N/A | N/A | 5.03 | 0.8616 | Random segment split | ❌ Leaky (Random split + Scaler) |
| `blood-pressure-2` | PPG | MAP / Avg | Linear Reg. | N/A | N/A | N/A | 0.0860 | Train on all / Random | ❌ Buggy code, failed model |
| `bloodpressure-abp` | PPG | ABP point | Dense MLP | N/A | N/A | 25.90 | N/A | Random point split | ❌ Broken formulation |
| `bloodpressure-anal.` | PPG + ECG | MAP, SBP, DBP | Random Forest | N/A | N/A | 13.28 | N/A | Random 5-fold K-Fold | ❌ Leaky split; ECG dependent |
| `cnn-lstm` | **PPG + ECG** | **SBP + DBP** | **CNN-LSTM** | **9.20** | **4.84** | **SBP: 13.50<br>DBP: 7.64** | **SBP: 0.645<br>DBP: 0.523** | **GroupShuffleSplit (by record)** | ✅ **Methodologically Sound (Dual signal)** |
| `comparing-models` | PPG | ABP wave | LR, MLP, LSTM | N/A | N/A | N/A | N/A | Random split | ⚠️ Cleared outputs |
| `ecg-ppg-ppg` | PPG + ECG | SBP + DBP | Random Forest | *(3.53)* | *(3.29)* | N/A | N/A | Random 5-fold | ❌ **Invalid (Index misalignment bug)** |
| `health-montoring` | Synthetic | Synthetic BP | SVR | *(7.89)* | N/A | N/A | N/A | Random split | ❌ **Invalid (Synthetic fake data)** |
| `ml-bp` | **PPG + ECG** | **$\Delta$SBP, $\Delta$DBP** | **XGBoost (Calibrated)** | **4.96** | **2.60** | **SBP: 7.81<br>DBP: 4.36** | **SBP: 0.873<br>DBP: 0.830** | **GroupKFold (by record)** | ⚠️ **Valid, but Calibration-Dependent** |
| `predict-min-max` | ECG only | SBP + DBP | ResNet-BiLSTM | 7.25 | 4.15 | N/A | N/A | Random 1% val split | ❌ Leaky split; No PPG |

---

## 16. Reusable Components

The following components represent high-quality engineering and should be adapted for the future system:

1. **ABP Ground Truth Generation (`extract_bp_from_abp` in `ml-bp.ipynb`)**:
   - Beat-by-beat peak detection using `find_peaks(abp, distance=50, prominence=10)` and inter-peak trough discovery.
   - Robustly calculates true mean SBP and DBP across windows while filtering out unphysiological pressure swings.
2. **Quality Control Filtering Pipeline (`cnn-lstm.ipynb` Cell 3)**:
   - Automated detection and removal of NaNs, flatline sensor dropouts ($\sigma < 10^{-6}$), physiological bounding ($50 \le \text{SBP} \le 250$, $30 \le \text{DBP} \le 150$), and PPG amplitude thresholds.
3. **Per-Segment Normalization (`normalize_per_segment` in `cnn-lstm.ipynb`)**:
   - Zero-leakage Z-score normalization computed strictly per segment window.
4. **Group-Aware Splitting Architecture**:
   - `GroupShuffleSplit` (`cnn-lstm.ipynb`) and `GroupKFold` (`ml-bp.ipynb`) grouped by `record_id` to guarantee zero patient overlap.
5. **PPG Morphological Feature Extraction (`ml-bp.ipynb` Cell 5)**:
   - The 13 PPG-only feature functions: VPG slope, crest time, systolic area, $b/a$ ratio, pulse width 50%, and spectral cardiac energy.
6. **Robust CNN-LSTM Baseline Architecture (`cnn-lstm.ipynb` Cell 15)**:
   - Conv1D feature extractor + LSTM temporal aggregator with batch normalization and dropout, modified to accept a single-channel PPG tensor `(250, 1)`.

---

## 17. Components That Should NOT Be Reused

1. **Point-by-Point Instantaneous Mapping (`bloodpressure-abp.ipynb`)**:
   - Attempting to predict blood pressure from individual unwindowed scalar points has zero physiological validity.
2. **Desynchronized Truncation Logic (`ecg-ppg-ppg.ipynb`)**:
   - Calling `.dropna()` on features while truncating targets by length (`[:len(X)]`).
3. **Synthetic Toy Data Generator (`health-montoring.ipynb`)**:
   - Random Gaussian numbers with arbitrary linear formulas.
4. **ECG-Dependent Features (PTT, PAT, ECG Hjorth, RR intervals)**:
   - Any code requiring Lead II ECG cannot function on an ESP32 + MAX30102 PPG device.
5. **Global Preprocessing Scalers Before Train/Test Split**:
   - StandardScalers fitted over full dataset arrays.
6. **Sub-second Windowing ($W = 64$ in `predict-min-max`)**:
   - Window sizes shorter than a single pulse period cannot capture systolic peaks and diastolic valleys.

---

## 18. Important Lessons From Existing Work

1. **The "Accuracy Illusion" of Random Splitting**:
   - Random window splitting inflates reported $R^2$ to $>0.95$ and drops MAE to $<3\text{ mmHg}$ purely because neighboring windows from the same subject are virtually identical. When evaluated on genuinely unseen records, true MAE rises to 8-10 mmHg.
2. **The Calibration Crutch**:
   - Models reporting $<5\text{ mmHg}$ MAE often secretly rely on having access to the patient's true arterial blood pressure at the start of inference. A truly autonomous wearable device cannot assume continuous invasive calibration.
3. **ECG is a Silent Dependency in Existing Literature**:
   - Most published "cuffless" algorithms rely heavily on Pulse Transit Time (PTT), which requires both ECG R-wave and PPG pulse arrival. Transitioning to PPG-only requires discovering blood pressure cues embedded entirely within pulse morphology, wave reflections, and harmonic resonance.
4. **Window Length Selection involves a Fundamental Trade-off**:
   - Windows shorter than 1.5 seconds fail to reliably capture full cardiac cycles across bradycardic rates.
   - 10-second windows capture 10-15 beats for stable feature extraction, but reduce sample count and temporal responsiveness.
   - 2.0 to 4.0 seconds (250 to 500 samples at 125 Hz) represents the sweet spot for deep learning continuous waveform models.

---

## 19. Open Questions for System Design

1. **Target Sampling Rate for Hardware Integration**:
   - The Kaggle MIMIC-II dataset is sampled at $125\text{ Hz}$. The MAX30102 sensor supports sampling rates such as $100\text{ Hz}, 200\text{ Hz}, 400\text{ Hz}$. Should our offline pipeline resample Kaggle to match the intended hardware sampling frequency (e.g. 100 Hz or retain 125 Hz)?
2. **Clinical Feasibility of Purely Calibration-Free PPG**:
   - Given that SBP MAE on unseen records without calibration hovers around 9.2 mmHg (failing AAMI STD $\le 8\text{ mmHg}$), should our future system support a practical one-time non-invasive cuff calibration (entering user baseline cuff reading once) or remain strictly calibration-free?
3. **Deep Learning vs Handcrafted Feature Engineering for Embedded Deployment**:
   - An end-to-end 1D-CNN + GRU/LSTM model running on raw waveforms avoids complex peak-detection edge cases, but requires floating-point inference (e.g. TensorFlow Lite Micro on ESP32 or offloading via Bluetooth/Wi-Fi to a local server/phone). Which deployment topology is preferred?

---

## 20. Recommendations for Next Phase

1. **Construct a Unified, Clean PPG-Only Data Extraction Pipeline**:
   - Load records across all available `.mat` files using Python (`h5py` / `scipy.io`).
   - Extract exclusively **Channel 0 (PPG)** and **Channel 1 (ABP)**; omit Channel 2 (ECG).
   - Apply the beat-by-beat physiological ground truth extractor (`find_peaks` on ABP with prominence $\ge 10\text{ mmHg}$) to generate gold-standard SBP and DBP targets.
2. **Implement Zero-Leakage Record-Independent Partitioning**:
   - Enforce `GroupShuffleSplit` or `GroupKFold` strictly by record index.
   - Under no circumstances allow segments from the same record to span across training, validation, and testing sets.
3. **Standardize PPG Signal Conditioning**:
   - Apply a zero-phase 3rd-order Butterworth bandpass filter ($[0.5, 8.0]\text{ Hz}$).
   - Use per-window Z-score normalization to ensure zero cross-window data leakage.
4. **Develop Dual Track Models for PPG-Only Inference**:
   - **Track A (Feature-Based Machine Learning)**: Extract the 13 PPG morphology and derivative features (VPG slope, crest time, $b/a$ ratio, stiffness proxy) and train LightGBM / XGBoost models.
   - **Track B (Deep Sequence Learning)**: Adapt the Conv1D-LSTM architecture from `cnn-lstm.ipynb` to accept a single-channel input tensor `(250, 1)` or `(500, 1)`.
5. **Evaluate Rigorously Against International Standards**:
   - Report Mean Error (Bias), Standard Deviation (STD), MAE, RMSE, and $R^2$ for SBP and DBP separately, benchmarked directly against the **AAMI SP10** standard and **British Hypertension Society (BHS)** grading criteria.
